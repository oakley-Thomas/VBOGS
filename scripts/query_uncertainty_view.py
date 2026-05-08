#!/usr/bin/env python3

"""Query a VBOGS uncertainty anchor map from one camera pose.

The script renders the normal Octree-AnyGS RGB image, renders the matching
per-anchor uncertainty heatmap, and exports anchors in the camera view cone with
their uncertainty values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json
from vbogs.render import render_scalar

DEFAULT_DRIVE = "2013_05_28_drive_0008_sync"
DEFAULT_OCTREE_OUTPUT_ROOTS = (Path("/data/OCTREE-ANYGS"), REPO_ROOT / "data" / "OCTREE-ANYGS")
ANCHOR_ROW_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("uncertainty", "<f4"),
        ("anchor_id", "<i4"),
        ("level", "<i2"),
        ("point_count", "<i4"),
        ("is_observed", "u1"),
        ("depth", "<f4"),
        ("pixel_x", "<f4"),
        ("pixel_y", "<f4"),
        ("in_frustum", "u1"),
        ("render_anchor_mask", "u1"),
        ("rendered_gaussian", "u1"),
    ]
)


@dataclass(frozen=True)
class ProjectionResult:
    camera_xyz: np.ndarray
    pixel_xy: np.ndarray
    in_frustum: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default=DEFAULT_DRIVE)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Octree-AnyGS run directory. Defaults to the latest run for --drive.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Directory with M4/M5 artifacts. Defaults to `data/m4/<drive>`.",
    )
    parser.add_argument(
        "--u-path",
        type=Path,
        default=None,
        help="Per-anchor uncertainty array. Defaults to `<bucket-root>/U.npy`.",
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=None,
        help="Optional posterior artifact for `is_observed`. Defaults to auto-detect.",
    )
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--octree-root", type=Path, default=Path("Octree-AnyGS"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to `outputs/uncertainty_queries/<drive>`.",
    )
    parser.add_argument("--name", default=None, help="Stem used for output filenames.")

    pose_group = parser.add_mutually_exclusive_group()
    pose_group.add_argument(
        "--pose",
        nargs="+",
        default=None,
        help=(
            "Camera pose as 16 row-major numbers. Quote the value or pass the "
            "16 numbers after --pose."
        ),
    )
    pose_group.add_argument(
        "--pose-file",
        type=Path,
        default=None,
        help="Text, JSON, .npy, or .npz file containing a 4x4 pose matrix.",
    )
    parser.add_argument(
        "--pose-convention",
        choices=("c2w", "w2c"),
        default="c2w",
        help="Convention for --pose or generic matrix files. Explicit JSON/NPZ keys win.",
    )
    parser.add_argument(
        "--camera-source",
        choices=("test", "train"),
        default="test",
        help="Existing Octree-AnyGS split to use when --pose/--pose-file is omitted.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Existing camera index to query when --pose/--pose-file is omitted.",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Optional existing camera image name/stem to query instead of --camera-index.",
    )
    parser.add_argument(
        "--reference-source",
        choices=("test", "train"),
        default="test",
        help="Split providing intrinsics/resolution for custom poses.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Camera index providing intrinsics/resolution for custom poses.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)

    parser.add_argument(
        "--anchor-filter",
        choices=("frustum", "render-mask", "rendered", "frustum-and-render-mask", "frustum-and-rendered", "all"),
        default="frustum",
        help="Which anchor rows to export. `frustum` is the geometric view cone.",
    )
    parser.add_argument("--colormap", default="turbo")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--percentile-low", type=float, default=2.0)
    parser.add_argument("--percentile-high", type=float, default=98.0)
    parser.add_argument("--alpha-threshold", type=float, default=1.0e-6)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--ape", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export for large queries.")
    parser.add_argument("--no-ply", action="store_true", help="Skip PLY export.")
    return parser.parse_args()


def split_numeric_tokens(values: Iterable[str]) -> list[float]:
    tokens: list[float] = []
    for value in values:
        for token in re.split(r"[\s,;]+", str(value).strip()):
            if token:
                tokens.append(float(token))
    return tokens


def safe_output_stem(value: str) -> str:
    stem = Path(str(value)).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe or "query"


def matrix_from_values(values: Iterable[str]) -> np.ndarray:
    numbers = split_numeric_tokens(values)
    if len(numbers) == 16:
        return np.asarray(numbers, dtype=np.float64).reshape(4, 4)
    if len(numbers) == 12:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :] = np.asarray(numbers, dtype=np.float64).reshape(3, 4)
        return matrix
    raise ValueError(f"Expected 16 values for a 4x4 matrix, or 12 values for a 3x4 matrix; got {len(numbers)}")


def load_pose_file(path: Path, fallback_convention: str) -> tuple[np.ndarray, str]:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return coerce_pose_matrix(np.load(path)), fallback_convention
    if suffix == ".npz":
        payload = np.load(path)
        for key in ("c2w", "camera_to_world"):
            if key in payload:
                return coerce_pose_matrix(payload[key]), "c2w"
        for key in ("w2c", "world_to_camera", "world_view_transform"):
            if key in payload:
                return coerce_pose_matrix(payload[key]), "w2c"
        if "matrix" in payload:
            return coerce_pose_matrix(payload["matrix"]), fallback_convention
        raise KeyError(f"{path} does not contain one of c2w, w2c, world_to_camera, or matrix")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("c2w", "camera_to_world"):
                if key in payload:
                    return coerce_pose_matrix(payload[key]), "c2w"
            for key in ("w2c", "world_to_camera", "world_view_transform"):
                if key in payload:
                    return coerce_pose_matrix(payload[key]), "w2c"
            if "matrix" in payload:
                convention = str(payload.get("convention", fallback_convention)).lower()
                if convention not in ("c2w", "w2c"):
                    raise ValueError(f"Unsupported pose convention in {path}: {convention}")
                return coerce_pose_matrix(payload["matrix"]), convention
        return coerce_pose_matrix(payload), fallback_convention
    return matrix_from_values(path.read_text(encoding="utf-8").split()), fallback_convention


def coerce_pose_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        padded = np.eye(4, dtype=np.float64)
        padded[:3, :] = matrix
        return padded
    if matrix.size in (12, 16):
        return matrix_from_values([str(item) for item in matrix.reshape(-1).tolist()])
    raise ValueError(f"Expected pose matrix shape (4, 4), (3, 4), 16, or 12 values; got {matrix.shape}")


def pose_to_c2w(matrix: np.ndarray, convention: str) -> np.ndarray:
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix, got {matrix.shape}")
    if convention == "c2w":
        c2w = matrix
    elif convention == "w2c":
        c2w = np.linalg.inv(matrix)
    else:
        raise ValueError(f"Unsupported pose convention: {convention}")
    if not np.all(np.isfinite(c2w)):
        raise ValueError("Pose matrix contains non-finite values")
    return c2w.astype(np.float32)


def add_octree_to_path(octree_root: Path) -> Path:
    octree_root = octree_root.resolve()
    if not octree_root.exists():
        raise FileNotFoundError(f"Octree-AnyGS root not found: {octree_root}")
    if str(octree_root) not in sys.path:
        sys.path.insert(0, str(octree_root))
    return octree_root


def resolve_model_path(drive: str, model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path.resolve()
    for root in DEFAULT_OCTREE_OUTPUT_ROOTS:
        drive_root = root / drive
        if not drive_root.exists():
            continue
        candidates = sorted(
            path for path in drive_root.iterdir() if path.is_dir() and (path / "config.yaml").exists()
        )
        if candidates:
            return candidates[-1].resolve()
    searched = ", ".join(str(root / drive) for root in DEFAULT_OCTREE_OUTPUT_ROOTS)
    raise FileNotFoundError(f"No Octree-AnyGS run found. Searched: {searched}")


def resolve_bucket_root(drive: str, bucket_root: Path | None) -> Path:
    if bucket_root is not None:
        return bucket_root.resolve()
    return (REPO_ROOT / "data" / "m4" / drive).resolve()


def resolve_u_path(bucket_root: Path, u_path: Path | None) -> Path:
    if u_path is not None:
        return u_path.resolve()
    return (bucket_root / "U.npy").resolve()


def resolve_posterior_path(bucket_root: Path, posterior: Path | None) -> Path | None:
    if posterior is not None:
        return posterior.resolve()
    for name in ("anchor_posterior.npz", "anchor_posterior.smoke.npz"):
        candidate = bucket_root / name
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_output_dir(drive: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (REPO_ROOT / "outputs" / "uncertainty_queries" / drive).resolve()


def load_octree_scene(model_path: Path, iteration: int, octree_root: Path, ape_code: int, quiet: bool):
    import yaml

    add_octree_to_path(octree_root)
    from scene import Scene
    from utils.general_utils import parse_cfg, safe_state

    with (model_path / "config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    dataset, opt, pipe = parse_cfg(cfg)
    dataset.model_path = str(model_path)

    model_config = dataset.model_config
    module_name = f"scene.{model_config['kwargs']['gs_attr'][:-2]}_model"
    module = __import__(module_name, fromlist=[""])
    gaussians = getattr(module, model_config["name"])(**model_config["kwargs"])
    gaussians.ape_code = int(ape_code)

    safe_state(quiet)
    scene = Scene(dataset, opt, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.eval()
    return scene, gaussians, pipe


def selected_cameras(scene: Any, source: str) -> list[Any]:
    cameras = scene.getTestCameras() if source == "test" else scene.getTrainCameras()
    return list(cameras)


def camera_by_name(cameras: list[Any], name: str) -> Any:
    wanted = Path(name).stem
    for cam in cameras:
        image_name = Path(str(getattr(cam, "image_name", ""))).stem
        if image_name == wanted:
            return cam
    raise ValueError(f"No camera named {name!r} found")


def camera_to_c2w(cam: Any) -> np.ndarray:
    matrix = cam.world_view_transform.transpose(0, 1).inverse()
    return matrix.detach().cpu().numpy().astype(np.float32)


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


class QueryCamera:
    """Small camera object compatible with Octree-AnyGS render functions."""

    def __init__(
        self,
        *,
        source_cam: Any,
        c2w: np.ndarray,
        uid: int,
        image_name: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        znear: float,
        zfar: float,
    ) -> None:
        import torch
        from utils.graphics_utils import getProjectionMatrix

        self.uid = int(uid)
        self.image_name = image_name
        self.image_path = getattr(source_cam, "image_path", "")
        self.resolution_scale = float(getattr(source_cam, "resolution_scale", 1.0))
        self.image_width = int(width)
        self.image_height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.FoVx = focal_to_fov(self.fx, self.image_width)
        self.FoVy = focal_to_fov(self.fy, self.image_height)
        self.znear = float(znear)
        self.zfar = float(zfar)

        self.c2w_np = np.asarray(c2w, dtype=np.float32)
        world_to_view = np.linalg.inv(self.c2w_np).astype(np.float32)
        self.world_view_transform = torch.tensor(world_to_view, dtype=torch.float32, device="cuda").transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()


def build_query_camera(
    *,
    source_cam: Any,
    c2w: np.ndarray,
    image_name: str,
    args: argparse.Namespace,
) -> QueryCamera:
    width = int(args.width or getattr(source_cam, "image_width"))
    height = int(args.height or getattr(source_cam, "image_height"))
    fx = float(args.fx if args.fx is not None else getattr(source_cam, "fx"))
    fy = float(args.fy if args.fy is not None else getattr(source_cam, "fy"))
    cx = float(args.cx if args.cx is not None else getattr(source_cam, "cx", width * 0.5))
    cy = float(args.cy if args.cy is not None else getattr(source_cam, "cy", height * 0.5))
    znear = float(args.near if args.near is not None else getattr(source_cam, "znear", 0.01))
    zfar = float(args.far if args.far is not None else getattr(source_cam, "zfar", 100.0))
    return QueryCamera(
        source_cam=source_cam,
        c2w=c2w,
        uid=int(getattr(source_cam, "uid", 0)),
        image_name=image_name,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        znear=znear,
        zfar=zfar,
    )


def resolve_camera(scene: Any, args: argparse.Namespace) -> Any:
    cameras = selected_cameras(scene, args.camera_source)
    if not cameras:
        raise ValueError(f"No {args.camera_source} cameras available")
    if args.camera_name:
        return camera_by_name(cameras, args.camera_name)
    if not 0 <= args.camera_index < len(cameras):
        raise IndexError(
            f"--camera-index {args.camera_index} is out of range for {len(cameras)} {args.camera_source} cameras"
        )
    return cameras[args.camera_index]


def resolve_reference_camera(scene: Any, args: argparse.Namespace) -> Any:
    cameras = selected_cameras(scene, args.reference_source)
    if not cameras:
        raise ValueError(f"No {args.reference_source} cameras available for custom pose intrinsics")
    index = min(max(args.reference_index, 0), len(cameras) - 1)
    return cameras[index]


def make_query_camera(scene: Any, args: argparse.Namespace) -> tuple[Any, np.ndarray, str]:
    if args.pose is None and args.pose_file is None:
        cam = resolve_camera(scene, args)
        name = safe_output_stem(
            args.name or str(getattr(cam, "image_name", f"{args.camera_source}_{args.camera_index:04d}"))
        )
        return cam, camera_to_c2w(cam), name

    if args.pose is not None:
        pose_matrix = matrix_from_values(args.pose)
        convention = args.pose_convention
    else:
        pose_matrix, convention = load_pose_file(args.pose_file, args.pose_convention)

    c2w = pose_to_c2w(pose_matrix, convention)
    reference = resolve_reference_camera(scene, args)
    name = safe_output_stem(args.name or "custom_pose")
    return build_query_camera(source_cam=reference, c2w=c2w, image_name=name, args=args), c2w, name


def load_anchor_artifacts(
    *,
    bucket_root: Path,
    u_path: Path,
    posterior_path: Path | None,
    anchor_count: int,
) -> dict[str, np.ndarray]:
    pts_path = bucket_root / "pts_by_anchor.npz"
    if not pts_path.exists():
        raise FileNotFoundError(f"Could not find anchor artifact: {pts_path}")
    if not u_path.exists():
        raise FileNotFoundError(f"Could not find uncertainty array: {u_path}")

    pts = np.load(pts_path)
    uncertainty = np.asarray(np.load(u_path), dtype=np.float32).reshape(-1)
    anchor_xyz = np.asarray(pts["anchor_xyz"], dtype=np.float32)
    anchor_level = np.asarray(pts["anchor_level"], dtype=np.int16).reshape(-1)
    point_counts = np.asarray(pts["point_counts"], dtype=np.int32).reshape(-1)

    if anchor_xyz.shape[0] != anchor_count:
        raise ValueError(f"pts_by_anchor has {anchor_xyz.shape[0]} anchors, scene has {anchor_count}")
    for name, values in (
        ("uncertainty", uncertainty),
        ("anchor_level", anchor_level),
        ("point_counts", point_counts),
    ):
        if values.shape[0] != anchor_count:
            raise ValueError(f"{name} length {values.shape[0]} does not match anchor count {anchor_count}")

    if posterior_path is not None and posterior_path.exists():
        posterior = np.load(posterior_path)
        is_observed = np.asarray(posterior["is_observed"], dtype=bool).reshape(-1)
        if is_observed.shape[0] != anchor_count:
            raise ValueError(
                f"is_observed length {is_observed.shape[0]} does not match anchor count {anchor_count}"
            )
    else:
        is_observed = point_counts > 0

    return {
        "anchor_xyz": anchor_xyz,
        "anchor_level": anchor_level,
        "point_counts": point_counts,
        "uncertainty": uncertainty,
        "is_observed": is_observed,
    }


def project_points_to_camera(
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float,
    far: float,
) -> ProjectionResult:
    points_world = np.asarray(points_world, dtype=np.float64)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"Expected points_world shape (N, 3), got {points_world.shape}")
    if world_to_camera.shape != (4, 4):
        raise ValueError(f"Expected world_to_camera shape (4, 4), got {world_to_camera.shape}")

    homogeneous = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    camera_h = (world_to_camera @ homogeneous.T).T
    camera_xyz = camera_h[:, :3]
    z = camera_xyz[:, 2]

    pixel_xy = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    valid_z = np.isfinite(z) & (np.abs(z) > 1.0e-12)
    pixel_xy[valid_z, 0] = (float(fx) * camera_xyz[valid_z, 0] / z[valid_z] + float(cx)).astype(np.float32)
    pixel_xy[valid_z, 1] = (float(fy) * camera_xyz[valid_z, 1] / z[valid_z] + float(cy)).astype(np.float32)

    in_frustum = (
        valid_z
        & np.isfinite(pixel_xy[:, 0])
        & np.isfinite(pixel_xy[:, 1])
        & (z >= float(near))
        & (z <= float(far))
        & (pixel_xy[:, 0] >= 0.0)
        & (pixel_xy[:, 0] < float(width))
        & (pixel_xy[:, 1] >= 0.0)
        & (pixel_xy[:, 1] < float(height))
    )
    return ProjectionResult(
        camera_xyz=camera_xyz.astype(np.float32),
        pixel_xy=pixel_xy,
        in_frustum=in_frustum,
    )


def rendered_anchor_mask_from_gaussians(
    visible_mask: np.ndarray,
    selection_mask: np.ndarray,
    visibility_filter: np.ndarray,
    n_offsets: int,
) -> np.ndarray:
    visible_mask = np.asarray(visible_mask, dtype=bool).reshape(-1)
    selection_mask = np.asarray(selection_mask, dtype=bool).reshape(-1)
    visibility_filter = np.asarray(visibility_filter, dtype=bool).reshape(-1)
    visible_anchor_ids = np.nonzero(visible_mask)[0]
    expanded_anchor_ids = np.repeat(visible_anchor_ids, int(n_offsets))
    if expanded_anchor_ids.shape[0] != selection_mask.shape[0]:
        raise ValueError(
            "selection_mask length does not match visible anchors expanded by n_offsets "
            f"({selection_mask.shape[0]} != {expanded_anchor_ids.shape[0]})"
        )
    selected_anchor_ids = expanded_anchor_ids[selection_mask]
    if selected_anchor_ids.shape[0] != visibility_filter.shape[0]:
        raise ValueError(
            "visibility_filter length does not match selected gaussian count "
            f"({visibility_filter.shape[0]} != {selected_anchor_ids.shape[0]})"
        )
    rendered = np.zeros_like(visible_mask, dtype=bool)
    rendered_ids = selected_anchor_ids[visibility_filter]
    rendered[rendered_ids] = True
    return rendered


def world_to_camera_from_query_camera(cam: Any) -> np.ndarray:
    matrix = cam.world_view_transform.transpose(0, 1)
    return matrix.detach().cpu().numpy().astype(np.float32)


def choose_scale(values: np.ndarray, vmin: float | None, vmax: float | None, low: float, high: float) -> tuple[float, float]:
    if (vmin is None) != (vmax is None):
        raise ValueError("Pass both --vmin and --vmax, or neither")
    if vmin is not None and vmax is not None:
        if not vmin < vmax:
            raise ValueError(f"Expected --vmin < --vmax, got {vmin} >= {vmax}")
        return float(vmin), float(vmax)
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.01, 1.0e-6)
        lo -= pad
        hi += pad
    return lo, hi


def heatmap_tensor(values: Any, alpha: Any, *, vmin: float, vmax: float, colormap_name: str) -> Any:
    import matplotlib.cm as cm
    import torch

    normalized = ((values - float(vmin)) / float(vmax - vmin)).clamp(0.0, 1.0)
    colormap = cm.get_cmap(colormap_name)
    mapped = colormap(normalized.detach().cpu().numpy())[..., :3]
    heatmap = torch.from_numpy(mapped.astype(np.float32)).permute(2, 0, 1)
    alpha_mask = alpha.detach().cpu() > 0
    heatmap[:, ~alpha_mask] = 0.0
    return heatmap.to(device=values.device)


def uncertainty_rgb(values: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    t = (np.asarray(values, dtype=np.float32) - np.float32(vmin)) / np.float32(vmax - vmin)
    t = np.clip(np.where(np.isfinite(t), t, 1.0), 0.0, 1.0)
    # Compact blue-cyan-yellow-red diagnostic ramp.
    rgb = np.stack(
        [
            np.clip(1.5 * t - 0.2, 0.0, 1.0),
            np.clip(1.5 - np.abs(3.0 * t - 1.5), 0.0, 1.0),
            np.clip(1.2 - 1.8 * t, 0.0, 1.0),
        ],
        axis=1,
    )
    return np.rint(rgb * 255.0).astype(np.uint8)


def build_anchor_filter(
    filter_name: str,
    *,
    in_frustum: np.ndarray,
    render_anchor_mask: np.ndarray,
    rendered_gaussian_mask: np.ndarray,
) -> np.ndarray:
    if filter_name == "frustum":
        return in_frustum
    if filter_name == "render-mask":
        return render_anchor_mask
    if filter_name == "rendered":
        return rendered_gaussian_mask
    if filter_name == "frustum-and-render-mask":
        return in_frustum & render_anchor_mask
    if filter_name == "frustum-and-rendered":
        return in_frustum & rendered_gaussian_mask
    if filter_name == "all":
        return np.ones_like(in_frustum, dtype=bool)
    raise ValueError(f"Unknown anchor filter: {filter_name}")


def build_anchor_rows(
    *,
    anchor_ids: np.ndarray,
    anchor_xyz: np.ndarray,
    anchor_level: np.ndarray,
    point_counts: np.ndarray,
    is_observed: np.ndarray,
    uncertainty: np.ndarray,
    projection: ProjectionResult,
    in_frustum: np.ndarray,
    render_anchor_mask: np.ndarray,
    rendered_gaussian_mask: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    rgb = uncertainty_rgb(uncertainty[anchor_ids], vmin=vmin, vmax=vmax)
    rows = np.empty(anchor_ids.shape[0], dtype=ANCHOR_ROW_DTYPE)
    rows["x"] = anchor_xyz[anchor_ids, 0].astype(np.float32)
    rows["y"] = anchor_xyz[anchor_ids, 1].astype(np.float32)
    rows["z"] = anchor_xyz[anchor_ids, 2].astype(np.float32)
    rows["red"] = rgb[:, 0]
    rows["green"] = rgb[:, 1]
    rows["blue"] = rgb[:, 2]
    rows["uncertainty"] = uncertainty[anchor_ids].astype(np.float32)
    rows["anchor_id"] = anchor_ids.astype(np.int32)
    rows["level"] = anchor_level[anchor_ids].astype(np.int16)
    rows["point_count"] = point_counts[anchor_ids].astype(np.int32)
    rows["is_observed"] = is_observed[anchor_ids].astype(np.uint8)
    rows["depth"] = projection.camera_xyz[anchor_ids, 2].astype(np.float32)
    rows["pixel_x"] = projection.pixel_xy[anchor_ids, 0].astype(np.float32)
    rows["pixel_y"] = projection.pixel_xy[anchor_ids, 1].astype(np.float32)
    rows["in_frustum"] = in_frustum[anchor_ids].astype(np.uint8)
    rows["render_anchor_mask"] = render_anchor_mask[anchor_ids].astype(np.uint8)
    rows["rendered_gaussian"] = rendered_gaussian_mask[anchor_ids].astype(np.uint8)
    return rows


def write_anchor_csv(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows.dtype.names or ())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name].item() for name in fieldnames})


def ply_header(vertex_count: int) -> bytes:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float uncertainty",
            "property int anchor_id",
            "property short level",
            "property int point_count",
            "property uchar is_observed",
            "property float depth",
            "property float pixel_x",
            "property float pixel_y",
            "property uchar in_frustum",
            "property uchar render_anchor_mask",
            "property uchar rendered_gaussian",
            "end_header",
            "",
        ]
    )
    return header.encode("ascii")


def write_anchor_ply(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(ply_header(int(rows.shape[0])))
        rows.astype(ANCHOR_ROW_DTYPE, copy=False).tofile(handle)


def save_query_images(
    *,
    output_dir: Path,
    stem: str,
    rgb: Any,
    unc_image: Any,
    alpha_image: Any,
    vmin: float,
    vmax: float,
    colormap: str,
    alpha_threshold: float,
    overlay_alpha: float,
) -> dict[str, str]:
    import torch
    import torchvision

    alpha_mask = alpha_image > float(alpha_threshold)
    display_unc = torch.zeros_like(unc_image)
    display_unc[alpha_mask] = unc_image[alpha_mask] / alpha_image[alpha_mask].clamp_min(1.0e-8)
    heatmap = heatmap_tensor(display_unc, alpha_image, vmin=vmin, vmax=vmax, colormap_name=colormap)
    heatmap = heatmap.to(rgb.device)
    overlay = torch.clamp((1.0 - overlay_alpha) * rgb + overlay_alpha * heatmap, 0.0, 1.0)
    overlay[:, ~alpha_mask] = rgb[:, ~alpha_mask]

    paths = {
        "rgb": output_dir / f"{stem}_rgb.png",
        "uncertainty_heatmap": output_dir / f"{stem}_uncertainty_heatmap.png",
        "uncertainty_overlay": output_dir / f"{stem}_uncertainty_overlay.png",
        "side_by_side": output_dir / f"{stem}_side_by_side.png",
        "uncertainty_accumulated_npy": output_dir / f"{stem}_uncertainty_accumulated.npy",
        "uncertainty_display_npy": output_dir / f"{stem}_uncertainty_display.npy",
        "alpha_npy": output_dir / f"{stem}_alpha.npy",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(rgb, str(paths["rgb"]))
    torchvision.utils.save_image(heatmap, str(paths["uncertainty_heatmap"]))
    torchvision.utils.save_image(overlay, str(paths["uncertainty_overlay"]))
    torchvision.utils.save_image(torch.cat([rgb, heatmap, overlay], dim=2), str(paths["side_by_side"]))
    np.save(paths["uncertainty_accumulated_npy"], unc_image.detach().cpu().numpy().astype(np.float32))
    np.save(paths["uncertainty_display_npy"], display_unc.detach().cpu().numpy().astype(np.float32))
    np.save(paths["alpha_npy"], alpha_image.detach().cpu().numpy().astype(np.float32))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.drive, args.model_path)
    bucket_root = resolve_bucket_root(args.drive, args.bucket_root)
    u_path = resolve_u_path(bucket_root, args.u_path)
    posterior_path = resolve_posterior_path(bucket_root, args.posterior)
    output_dir = resolve_output_dir(args.drive, args.output_dir)

    print(f"Loading Octree-AnyGS model: {model_path}")
    scene, gaussians, pipe = load_octree_scene(model_path, args.iteration, args.octree_root, args.ape, args.quiet)
    loaded_iteration = int(scene.loaded_iter)
    anchor_count = int(gaussians.get_anchor.shape[0])

    query_cam, c2w, stem = make_query_camera(scene, args)
    print(f"Query camera: {stem}")
    print(f"Loading uncertainty: {u_path}")
    anchors = load_anchor_artifacts(
        bucket_root=bucket_root,
        u_path=u_path,
        posterior_path=posterior_path,
        anchor_count=anchor_count,
    )

    import torch
    from gaussian_renderer.render import render as render_rgb

    uncertainty_t = torch.from_numpy(anchors["uncertainty"].astype(np.float32, copy=False)).to(
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        rgb_pkg = render_rgb(query_cam, gaussians, pipe, scene.background, loaded_iteration)
        rgb = torch.clamp(rgb_pkg["render"], 0.0, 1.0)
        scalar_result = render_scalar(query_cam, gaussians, pipe, uncertainty_t, loaded_iteration)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    unc_image = scalar_result["unc_image"]
    alpha_image = scalar_result["alpha_image"]
    alpha_mask = alpha_image > args.alpha_threshold
    if bool(alpha_mask.any().item()):
        display_unc_np = torch.zeros_like(unc_image)
        display_unc_np[alpha_mask] = unc_image[alpha_mask] / alpha_image[alpha_mask].clamp_min(1.0e-8)
        scale_values = display_unc_np[alpha_mask].detach().cpu().numpy()
    else:
        scale_values = anchors["uncertainty"]
    vmin, vmax = choose_scale(scale_values, args.vmin, args.vmax, args.percentile_low, args.percentile_high)

    image_paths = save_query_images(
        output_dir=output_dir,
        stem=stem,
        rgb=rgb,
        unc_image=unc_image,
        alpha_image=alpha_image,
        vmin=vmin,
        vmax=vmax,
        colormap=args.colormap,
        alpha_threshold=args.alpha_threshold,
        overlay_alpha=args.overlay_alpha,
    )

    projection = project_points_to_camera(
        anchors["anchor_xyz"],
        world_to_camera_from_query_camera(query_cam),
        fx=float(query_cam.fx),
        fy=float(query_cam.fy),
        cx=float(query_cam.cx),
        cy=float(query_cam.cy),
        width=int(query_cam.image_width),
        height=int(query_cam.image_height),
        near=float(query_cam.znear),
        far=float(query_cam.zfar),
    )

    render_anchor_mask = scalar_result["visible_mask"].detach().cpu().numpy().astype(bool)
    rendered_gaussian_mask = rendered_anchor_mask_from_gaussians(
        render_anchor_mask,
        scalar_result["selection_mask"].detach().cpu().numpy().astype(bool),
        scalar_result["visibility_filter"].detach().cpu().numpy().astype(bool),
        int(gaussians.n_offsets),
    )
    export_mask = build_anchor_filter(
        args.anchor_filter,
        in_frustum=projection.in_frustum,
        render_anchor_mask=render_anchor_mask,
        rendered_gaussian_mask=rendered_gaussian_mask,
    )
    anchor_ids = np.nonzero(export_mask)[0].astype(np.int64)
    rows = build_anchor_rows(
        anchor_ids=anchor_ids,
        anchor_xyz=anchors["anchor_xyz"],
        anchor_level=anchors["anchor_level"],
        point_counts=anchors["point_counts"],
        is_observed=anchors["is_observed"],
        uncertainty=anchors["uncertainty"],
        projection=projection,
        in_frustum=projection.in_frustum,
        render_anchor_mask=render_anchor_mask,
        rendered_gaussian_mask=rendered_gaussian_mask,
        vmin=vmin,
        vmax=vmax,
    )

    anchor_npz_path = output_dir / f"{stem}_anchors.npz"
    np.savez_compressed(
        anchor_npz_path,
        anchor_id=rows["anchor_id"],
        xyz=np.stack([rows["x"], rows["y"], rows["z"]], axis=1).astype(np.float32),
        uncertainty=rows["uncertainty"],
        level=rows["level"],
        point_count=rows["point_count"],
        is_observed=rows["is_observed"].astype(bool),
        depth=rows["depth"],
        pixel_xy=np.stack([rows["pixel_x"], rows["pixel_y"]], axis=1).astype(np.float32),
        in_frustum=rows["in_frustum"].astype(bool),
        render_anchor_mask=rows["render_anchor_mask"].astype(bool),
        rendered_gaussian=rows["rendered_gaussian"].astype(bool),
    )

    csv_path = None
    if not args.no_csv:
        csv_path = output_dir / f"{stem}_anchors.csv"
        write_anchor_csv(csv_path, rows)

    ply_path = None
    if not args.no_ply:
        ply_path = output_dir / f"{stem}_anchors.ply"
        write_anchor_ply(ply_path, rows)

    finite_u = rows["uncertainty"][np.isfinite(rows["uncertainty"])]
    metadata = {
        "drive": args.drive,
        "model_path": str(model_path),
        "bucket_root": str(bucket_root),
        "u_path": str(u_path),
        "posterior_path": str(posterior_path) if posterior_path is not None else None,
        "iteration": loaded_iteration,
        "query_name": stem,
        "pose_convention_output": "c2w",
        "c2w": c2w.astype(float).tolist(),
        "intrinsics": {
            "width": int(query_cam.image_width),
            "height": int(query_cam.image_height),
            "fx": float(query_cam.fx),
            "fy": float(query_cam.fy),
            "cx": float(query_cam.cx),
            "cy": float(query_cam.cy),
            "znear": float(query_cam.znear),
            "zfar": float(query_cam.zfar),
        },
        "anchor_filter": args.anchor_filter,
        "anchor_count_total": anchor_count,
        "anchor_count_exported": int(rows.shape[0]),
        "anchor_count_in_frustum": int(np.count_nonzero(projection.in_frustum)),
        "anchor_count_render_mask": int(np.count_nonzero(render_anchor_mask)),
        "anchor_count_rendered_gaussian": int(np.count_nonzero(rendered_gaussian_mask)),
        "alpha_sum": float(alpha_image.sum().detach().cpu()),
        "uncertainty_accumulated_sum": float(unc_image.sum().detach().cpu()),
        "uncertainty_display_scale": {
            "vmin": vmin,
            "vmax": vmax,
            "percentile_low": args.percentile_low,
            "percentile_high": args.percentile_high,
            "colormap": args.colormap,
        },
        "exported_uncertainty_summary": {
            "min": float(np.min(finite_u)) if finite_u.size else None,
            "p50": float(np.percentile(finite_u, 50)) if finite_u.size else None,
            "p90": float(np.percentile(finite_u, 90)) if finite_u.size else None,
            "max": float(np.max(finite_u)) if finite_u.size else None,
        },
        "outputs": {
            **image_paths,
            "anchors_npz": str(anchor_npz_path),
            "anchors_csv": str(csv_path) if csv_path is not None else None,
            "anchors_ply": str(ply_path) if ply_path is not None else None,
        },
    }
    metadata_path = output_dir / f"{stem}_metadata.json"
    save_json(metadata_path, metadata)

    print(f"Wrote RGB/heatmap diagnostics under {output_dir}")
    print(f"Wrote {anchor_npz_path}")
    if csv_path is not None:
        print(f"Wrote {csv_path}")
    if ply_path is not None:
        print(f"Wrote {ply_path}")
    print(f"Wrote {metadata_path}")
    print(
        f"Anchors exported: {rows.shape[0]:,} | "
        f"in_frustum={metadata['anchor_count_in_frustum']:,} | "
        f"rendered_gaussian={metadata['anchor_count_rendered_gaussian']:,}"
    )


if __name__ == "__main__":
    main()
