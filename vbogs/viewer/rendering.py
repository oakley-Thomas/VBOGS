"""Server-side rendering runtime for the realtime debug viewer."""

from __future__ import annotations

import io
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vbogs.octree_config import load_octree_config_for_model
from vbogs.render import render_scalar
from vbogs.viewer.camera import LightweightSceneCamera, ViewerCamera, camera_to_c2w
from vbogs.viewer.pose import pose_to_c2w, request_payload_to_c2w

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRIVE = "2013_05_28_drive_0008_sync"
DEFAULT_OCTREE_OUTPUT_ROOTS = (Path("/data/OCTREE-ANYGS"), REPO_ROOT / "data" / "OCTREE-ANYGS")
DEFAULT_VIEWER_PORT = 8070
RENDER_MODES = {"rgb", "uncertainty", "alpha", "side_by_side"}


@dataclass(frozen=True)
class RenderedFrame:
    metadata: dict[str, Any]
    jpeg: bytes


@dataclass(frozen=True)
class ResolvedViewerCamera:
    entry: "CameraEntry"
    c2w: np.ndarray
    camera: ViewerCamera


@dataclass(frozen=True)
class CameraEntry:
    source: str
    index: int
    camera: Any
    c2w: np.ndarray

    @property
    def camera_id(self) -> str:
        return f"{self.source}:{self.index}"

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.camera_id,
            "source": self.source,
            "index": self.index,
            "image_name": str(getattr(self.camera, "image_name", f"{self.source}_{self.index:04d}")),
            "width": int(getattr(self.camera, "image_width")),
            "height": int(getattr(self.camera, "image_height")),
            "fx": float(getattr(self.camera, "fx")),
            "fy": float(getattr(self.camera, "fy")),
            "cx": float(getattr(self.camera, "cx", float(getattr(self.camera, "image_width")) * 0.5)),
            "cy": float(getattr(self.camera, "cy", float(getattr(self.camera, "image_height")) * 0.5)),
            "c2w": self.c2w.astype(float).tolist(),
        }


def add_octree_to_path(octree_root: Path) -> Path:
    octree_root = octree_root.resolve()
    if not octree_root.exists():
        raise FileNotFoundError(f"Octree-AnyGS root not found: {octree_root}")
    if str(octree_root) not in sys.path:
        sys.path.insert(0, str(octree_root))
    return octree_root


def resolve_model_path(drive: str, model_path: Path | None) -> Path:
    if model_path is not None:
        model_path = model_path.resolve()
        if not (model_path / "config.yaml").exists():
            raise FileNotFoundError(f"Octree-AnyGS config not found: {model_path / 'config.yaml'}")
        return model_path

    searched = []
    for root in DEFAULT_OCTREE_OUTPUT_ROOTS:
        drive_root = root / drive
        searched.append(str(drive_root))
        if not drive_root.exists():
            continue
        candidates = sorted(path for path in drive_root.glob("*") if path.is_dir() and (path / "config.yaml").exists())
        if candidates:
            return candidates[-1].resolve()
    raise FileNotFoundError(f"No Octree-AnyGS run found. Searched: {searched}")


def resolve_uncertainty_path(drive: str, u_path: Path | None) -> Path:
    if u_path is not None:
        return u_path.resolve()
    return (REPO_ROOT / "data" / "m4" / drive / "U.npy").resolve()


def read_colmap_cameras_metadata_only(
    cam_extrinsics: dict[Any, Any],
    cam_intrinsics: dict[Any, Any],
    depths_params: dict[str, Any] | None,
    images_folder: str,
    masks_folder: str | None,
    depths_folder: str | None,
) -> list[Any]:
    """Build CameraInfo records without opening image files.

    Octree-AnyGS keeps PIL handles and later uploads every source image to CUDA.
    That behavior is useful for training/evaluation losses, but the realtime
    viewer only needs camera metadata to render the trained scene.
    """

    if masks_folder is not None or depths_folder is not None or depths_params is not None:
        raise ValueError("Metadata-only viewer loading does not support masks or depth maps")

    from scene.colmap_loader import qvec2rotmat
    from scene.dataset_readers import CameraInfo
    from utils.graphics_utils import focal2fov
    import os

    cam_infos = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = int(intr.height)
        width = int(intr.width)
        rotation = np.transpose(qvec2rotmat(extr.qvec))
        translation = np.asarray(extr.tvec)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            focal_length_x = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
            fovy = focal2fov(focal_length_x, height)
            fovx = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
            fovy = focal2fov(focal_length_y, height)
            fovx = focal2fov(focal_length_x, width)
        else:
            raise ValueError(f"Unsupported COLMAP camera model for viewer: {intr.model}")

        image_path = os.path.join(images_folder, extr.name)
        cam_infos.append(
            CameraInfo(
                uid=intr.id,
                R=rotation,
                T=translation,
                FovY=fovy,
                FovX=fovx,
                CX=cx,
                CY=cy,
                image=None,
                mask=None,
                depth=None,
                depth_params=None,
                image_path=image_path,
                image_name=os.path.basename(extr.name).split(".")[0],
                width=width,
                height=height,
            )
        )

    return sorted(cam_infos, key=lambda item: item.image_path)


def lightweight_camera_list_from_cam_infos(
    cam_infos: list[Any],
    resolution_scale: float,
    args: Any,
    background: Any,
) -> list[Any]:
    # Octree-AnyGS applies args.data_device only to image payloads; camera
    # matrices always live on CUDA (scene/cameras.py). Lightweight cameras
    # carry no image data, so a config trained with data_device=cpu must not
    # pull the pose tensors onto the CPU.
    return [
        LightweightSceneCamera(
            cam_info,
            uid=index,
            resolution_scale=resolution_scale,
            resolution_arg=args.resolution,
            device="cuda",
        )
        for index, cam_info in enumerate(cam_infos)
    ]


def install_lightweight_octree_camera_loader() -> None:
    """Patch Octree-AnyGS imports for viewer-only metadata cameras."""

    import scene as scene_module
    import scene.dataset_readers as dataset_readers

    dataset_readers.readColmapCameras = read_colmap_cameras_metadata_only
    scene_module.cameraList_from_camInfos = lightweight_camera_list_from_cam_infos


def validate_uncertainty_array(values: np.ndarray, anchor_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D uncertainty array, got shape {values.shape}")
    if values.shape[0] != int(anchor_count):
        raise ValueError(f"U.npy has {values.shape[0]} values, but the scene has {anchor_count} anchors")
    return values


def choose_scale_np(values: np.ndarray, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Uncertainty array contains no finite values")

    lo = float(np.percentile(finite, 2.0)) if vmin is None else float(vmin)
    hi = float(np.percentile(finite, 98.0)) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Invalid uncertainty color scale: vmin={lo}, vmax={hi}")
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalized_uncertainty(unc_image: Any, alpha_image: Any, *, alpha_threshold: float = 1.0e-8) -> Any:
    import torch

    values = torch.zeros_like(unc_image)
    mask = alpha_image > alpha_threshold
    values[mask] = unc_image[mask] / alpha_image[mask].clamp_min(alpha_threshold)
    return values


def _turbo_like(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * x - 1.5), 0.0, 1.0)
    b = np.clip(1.2 - 1.8 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def heatmap_tensor(values: Any, alpha: Any, *, vmin: float, vmax: float, colormap_name: str) -> Any:
    import torch

    alpha_mask = alpha > 0
    normalized = torch.zeros_like(values)
    normalized[alpha_mask] = (values[alpha_mask] - float(vmin)) / (float(vmax) - float(vmin))
    normalized = normalized.clamp(0.0, 1.0)
    normalized_np = normalized.detach().cpu().numpy()

    try:
        from matplotlib import colormaps

        mapped = colormaps.get_cmap(colormap_name)(normalized_np)[..., :3]
    except Exception:
        mapped = _turbo_like(normalized_np)

    heatmap = torch.from_numpy(mapped.astype(np.float32)).permute(2, 0, 1)
    heatmap[:, ~alpha_mask.detach().cpu()] = 0.0
    return heatmap.to(device=values.device)


def alpha_tensor(alpha: Any) -> Any:
    return alpha.clamp(0.0, 1.0)[None, ...].repeat(3, 1, 1)


def compose_layer(
    *,
    mode: str,
    rgb: Any | None,
    unc_image: Any | None,
    alpha_image: Any | None,
    vmin: float,
    vmax: float,
    colormap_name: str,
) -> Any:
    import torch

    if mode not in RENDER_MODES:
        raise ValueError(f"Unsupported render mode: {mode}")
    if mode == "rgb":
        if rgb is None:
            raise ValueError("RGB render is required for mode=rgb")
        return rgb
    if unc_image is None or alpha_image is None:
        raise ValueError(f"Uncertainty render is required for mode={mode}")

    display_unc = normalized_uncertainty(unc_image, alpha_image)
    heatmap = heatmap_tensor(display_unc, alpha_image, vmin=vmin, vmax=vmax, colormap_name=colormap_name)
    if mode == "uncertainty":
        return heatmap
    if mode == "alpha":
        return alpha_tensor(alpha_image)
    if rgb is None:
        raise ValueError("RGB render is required for mode=side_by_side")
    return torch.cat([rgb, heatmap], dim=2)


def tensor_to_jpeg(image_tensor: Any, *, quality: int) -> bytes:
    image_tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Expected image tensor shape (3, H, W), got {tuple(image_tensor.shape)}")
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image_np, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def resolve_request_c2w(payload: dict[str, Any], *, default_c2w: np.ndarray) -> np.ndarray:
    return request_payload_to_c2w(payload, default_c2w=default_c2w)


def gaussian_visibility_vector(visibility_filter: Any, gaussian_count: int) -> np.ndarray:
    values = np.asarray(visibility_filter, dtype=bool)
    if values.size == gaussian_count:
        return values.reshape(-1)
    values = np.squeeze(values)
    if values.size == gaussian_count:
        return values.reshape(-1)
    if values.ndim > 1 and values.shape[0] == gaussian_count:
        return np.any(values, axis=tuple(range(1, values.ndim))).reshape(-1)
    if values.ndim > 1 and values.shape[-1] == gaussian_count:
        return np.any(values, axis=tuple(range(values.ndim - 1))).reshape(-1)
    return values.reshape(-1)


def rendered_anchor_ids_from_gaussians(
    visible_mask: Any,
    selection_mask: Any,
    visibility_filter: Any,
    n_offsets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map visible rasterized gaussians back to unique parent anchor ids."""

    visible_mask = np.asarray(visible_mask, dtype=bool).reshape(-1)
    selection_mask = np.asarray(selection_mask, dtype=bool).reshape(-1)
    visible_anchor_ids = np.nonzero(visible_mask)[0].astype(np.int64)
    expanded_anchor_ids = np.repeat(visible_anchor_ids, int(n_offsets))
    if expanded_anchor_ids.shape[0] != selection_mask.shape[0]:
        raise ValueError(
            "selection_mask length does not match visible anchors expanded by n_offsets "
            f"({selection_mask.shape[0]} != {expanded_anchor_ids.shape[0]})"
        )
    selected_anchor_ids = expanded_anchor_ids[selection_mask]
    rendered_gaussians = gaussian_visibility_vector(visibility_filter, selected_anchor_ids.shape[0])
    if selected_anchor_ids.shape[0] != rendered_gaussians.shape[0]:
        raise ValueError(
            "visibility_filter length does not match selected gaussian count "
            f"({rendered_gaussians.shape[0]} != {selected_anchor_ids.shape[0]})"
        )
    rendered_anchor_ids, rendered_counts = np.unique(selected_anchor_ids[rendered_gaussians], return_counts=True)
    return rendered_anchor_ids.astype(np.int64), rendered_counts.astype(np.int64)


class OctreeRenderSession:
    """Load and render one Octree-AnyGS scene for browser clients."""

    def __init__(
        self,
        *,
        drive: str = DEFAULT_DRIVE,
        model_path: Path | None = None,
        u_path: Path | None = None,
        octree_root: Path = Path("Octree-AnyGS"),
        iteration: int = -1,
        resolution: int | None = 4,
        camera_source: str = "test",
        camera_index: int = 0,
        colormap: str = "turbo",
        vmin: float | None = None,
        vmax: float | None = None,
        force_all_levels: bool = False,
        rgb_only: bool = False,
        load_source_images: bool = False,
        jpeg_quality: int = 85,
        max_fps: float = 20.0,
        initial_c2w: np.ndarray | None = None,
    ) -> None:
        if camera_source not in ("train", "test"):
            raise ValueError("--camera-source must be `train` or `test`")
        if resolution is not None and int(resolution) == 0:
            raise ValueError("--resolution must be non-zero")

        self.drive = drive
        self.model_path = resolve_model_path(drive, model_path)
        self.octree_root = add_octree_to_path(octree_root)
        self.iteration = int(iteration)
        self.resolution = int(resolution) if resolution is not None else None
        self.colormap = colormap
        self.force_all_levels = bool(force_all_levels)
        self.rgb_only = bool(rgb_only)
        self.load_source_images = bool(load_source_images)
        self.jpeg_quality = int(jpeg_quality)
        self.max_fps = float(max_fps)
        self.initial_c2w = None if initial_c2w is None else pose_to_c2w(initial_c2w, "c2w")
        self.render_lock = threading.Lock()

        self.scene, self.gaussians, self.pipe = self._load_scene()
        self.loaded_iteration = int(self.scene.loaded_iter)
        self.anchor_count = int(self.gaussians.get_anchor.shape[0])
        self.cameras = self._collect_cameras()
        self.default_camera_id = self._default_camera_id(camera_source, camera_index)

        self.u_path = resolve_uncertainty_path(drive, u_path) if not self.rgb_only else None
        self.uncertainty = None
        self.vmin = None
        self.vmax = None
        if not self.rgb_only:
            if self.u_path is None or not self.u_path.exists():
                raise FileNotFoundError(f"Uncertainty file not found: {self.u_path}")
            u_values = validate_uncertainty_array(np.load(self.u_path), self.anchor_count)
            self.vmin, self.vmax = choose_scale_np(u_values, vmin, vmax)
            import torch

            self.uncertainty = torch.from_numpy(u_values).to(device="cuda", dtype=torch.float32)

    def _load_scene(self):
        from scene import Scene
        from utils.general_utils import parse_cfg, safe_state

        if not self.load_source_images:
            install_lightweight_octree_camera_loader()

        cfg = load_octree_config_for_model(self.model_path)
        dataset, opt, pipe = parse_cfg(cfg)
        if self.resolution is not None:
            dataset.resolution = self.resolution
        dataset.model_path = str(self.model_path)

        model_config = dataset.model_config
        module_name = "scene." + model_config["kwargs"]["gs_attr"][:-2] + "_model"
        module = __import__(module_name, fromlist=[""])
        gaussians = getattr(module, model_config["name"])(**model_config["kwargs"])
        gaussians.ape_code = -1

        safe_state(False)
        scene = Scene(dataset, opt, gaussians, load_iteration=self.iteration, shuffle=False)
        gaussians.eval()
        return scene, gaussians, pipe

    def _collect_cameras(self) -> dict[str, list[CameraEntry]]:
        cameras: dict[str, list[CameraEntry]] = {}
        for source, values in (("train", self.scene.getTrainCameras()), ("test", self.scene.getTestCameras())):
            entries = []
            for index, camera in enumerate(list(values)):
                entries.append(CameraEntry(source=source, index=index, camera=camera, c2w=camera_to_c2w(camera)))
            cameras[source] = entries
        return cameras

    def _default_camera_id(self, source: str, index: int) -> str:
        entries = self.cameras[source]
        if not entries:
            raise ValueError(f"No {source} cameras available")
        if not 0 <= int(index) < len(entries):
            raise IndexError(f"--camera-index {index} is out of range for {len(entries)} {source} cameras")
        return entries[int(index)].camera_id

    def metadata(self) -> dict[str, Any]:
        return {
            "drive": self.drive,
            "model_path": str(self.model_path),
            "u_path": str(self.u_path) if self.u_path is not None else None,
            "iteration": self.loaded_iteration,
            "resolution": self.resolution,
            "anchor_count": self.anchor_count,
            "rgb_only": self.rgb_only,
            "load_source_images": self.load_source_images,
            "render_modes": ["rgb"] if self.rgb_only else ["rgb", "uncertainty", "alpha", "side_by_side"],
            "default_camera_id": self.default_camera_id,
            "colormap": self.colormap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "max_fps": self.max_fps,
            "jpeg_quality": self.jpeg_quality,
            "initial_c2w": None if self.initial_c2w is None else self.initial_c2w.astype(float).tolist(),
        }

    def camera_payload(self) -> dict[str, Any]:
        return {
            "default_camera_id": self.default_camera_id,
            "cameras": {
                source: [entry.summary() for entry in entries]
                for source, entries in self.cameras.items()
            },
        }

    def _entry_by_id(self, camera_id: str | None) -> CameraEntry:
        wanted = camera_id or self.default_camera_id
        try:
            source, raw_index = wanted.split(":", 1)
            index = int(raw_index)
        except Exception as exc:
            raise ValueError(f"Invalid camera id: {wanted!r}") from exc
        if source not in self.cameras:
            raise ValueError(f"Unknown camera source: {source}")
        entries = self.cameras[source]
        if not 0 <= index < len(entries):
            raise IndexError(f"Camera id {wanted!r} is out of range")
        return entries[index]

    def _viewer_camera_from_payload(self, payload: dict[str, Any]) -> ResolvedViewerCamera:
        entry = self._entry_by_id(payload.get("camera_id"))
        c2w = resolve_request_c2w(payload, default_c2w=entry.c2w)
        viewer_cam = ViewerCamera.from_source(
            entry.camera,
            c2w=c2w,
            uid=entry.index,
            image_name=str(getattr(entry.camera, "image_name", entry.camera_id)),
            device="cuda",
        )
        return ResolvedViewerCamera(entry=entry, c2w=c2w, camera=viewer_cam)

    def render_request(self, payload: dict[str, Any]) -> RenderedFrame:
        request_id = payload.get("request_id")
        mode = str(payload.get("layer", payload.get("mode", "side_by_side")))
        if self.rgb_only and mode != "rgb":
            raise ValueError("This viewer was started with --rgb-only; only mode=rgb is available")
        if mode not in RENDER_MODES:
            raise ValueError(f"Unsupported render mode: {mode}")
        quality = int(payload.get("quality", self.jpeg_quality))
        quality = max(1, min(95, quality))

        resolved = self._viewer_camera_from_payload(payload)

        with self.render_lock:
            frame = self._render_frame(resolved.camera, mode=mode, quality=quality)
        frame.metadata["request_id"] = request_id
        frame.metadata["camera_id"] = resolved.entry.camera_id
        frame.metadata["pose_convention_output"] = "c2w"
        frame.metadata["c2w"] = resolved.c2w.astype(float).tolist()
        return frame

    def rendered_anchors_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.rgb_only or self.uncertainty is None:
            raise ValueError("This viewer was started with --rgb-only; rendered anchor uncertainty is unavailable")

        import torch

        request_id = payload.get("request_id")
        resolved = self._viewer_camera_from_payload(payload)
        with self.render_lock:
            with torch.no_grad():
                scalar_result = render_scalar(
                    resolved.camera,
                    self.gaussians,
                    self.pipe,
                    self.uncertainty,
                    self.loaded_iteration,
                    force_all_levels=self.force_all_levels,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        visible_mask = scalar_result["visible_mask"].detach().cpu().numpy().astype(bool)
        selection_mask = scalar_result["selection_mask"].detach().cpu().numpy().astype(bool)
        visibility_filter = scalar_result["visibility_filter"].detach().cpu().numpy().astype(bool)
        anchor_ids, gaussian_counts = rendered_anchor_ids_from_gaussians(
            visible_mask,
            selection_mask,
            visibility_filter,
            int(self.gaussians.n_offsets),
        )
        total_rendered_anchor_count = int(anchor_ids.shape[0])

        uncertainty_np = self.uncertainty.detach().cpu().numpy().astype(np.float32)
        all_rendered_uncertainty_values = (
            uncertainty_np[anchor_ids] if anchor_ids.size else np.empty((0,), dtype=np.float32)
        )
        anchor_xyz = self.gaussians.get_anchor.detach().cpu().numpy().astype(np.float32)
        anchor_level = None
        if hasattr(self.gaussians, "get_level"):
            anchor_level = self.gaussians.get_level.detach().cpu().numpy().reshape(-1)
        elif hasattr(self.gaussians, "_level"):
            anchor_level = self.gaussians._level.detach().cpu().numpy().reshape(-1)

        max_anchors = payload.get("max_anchors")
        truncated = False
        if max_anchors is not None:
            max_anchors = max(0, int(max_anchors))
            truncated = anchor_ids.shape[0] > max_anchors
            anchor_ids = anchor_ids[:max_anchors]
            gaussian_counts = gaussian_counts[:max_anchors]

        anchors = []
        for anchor_id, rendered_gaussian_count in zip(anchor_ids.tolist(), gaussian_counts.tolist()):
            row = {
                "anchor_id": int(anchor_id),
                "xyz": [float(x) for x in anchor_xyz[int(anchor_id)].tolist()],
                "uncertainty": float(uncertainty_np[int(anchor_id)]),
                "rendered_gaussian_count": int(rendered_gaussian_count),
            }
            if anchor_level is not None and int(anchor_id) < anchor_level.shape[0]:
                row["level"] = int(anchor_level[int(anchor_id)])
            anchors.append(row)

        unc_image_sum = float(scalar_result["unc_image"].sum().detach().cpu())
        alpha_sum = float(scalar_result["alpha_image"].sum().detach().cpu())
        return {
            "request_id": request_id,
            "camera_id": resolved.entry.camera_id,
            "pose_convention_output": "c2w",
            "c2w": resolved.c2w.astype(float).tolist(),
            "anchor_count_rendered": total_rendered_anchor_count,
            "anchor_count_returned": int(anchor_ids.shape[0]),
            "anchor_count_total_rendered_before_limit": total_rendered_anchor_count,
            "truncated": truncated,
            "total_anchor_uncertainty": (
                float(all_rendered_uncertainty_values.sum()) if all_rendered_uncertainty_values.size else 0.0
            ),
            "mean_anchor_uncertainty": (
                float(all_rendered_uncertainty_values.mean()) if all_rendered_uncertainty_values.size else None
            ),
            "uncertainty_image_sum": unc_image_sum,
            "alpha_sum": alpha_sum,
            "alpha_normalized_uncertainty": unc_image_sum / max(alpha_sum, 1.0e-8),
            "anchors": anchors,
        }

    def _render_frame(self, camera: ViewerCamera, *, mode: str, quality: int) -> RenderedFrame:
        import torch
        from gaussian_renderer.render import render as render_rgb

        started = time.perf_counter()
        rgb = None
        unc_image = None
        alpha_image = None

        with torch.no_grad():
            if mode in ("rgb", "side_by_side"):
                rgb_pkg = render_rgb(camera, self.gaussians, self.pipe, self.scene.background, self.loaded_iteration)
                rgb = torch.clamp(rgb_pkg["render"], 0.0, 1.0)
            if mode in ("uncertainty", "alpha", "side_by_side"):
                scalar_result = render_scalar(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.uncertainty,
                    self.loaded_iteration,
                    force_all_levels=self.force_all_levels,
                )
                unc_image = scalar_result["unc_image"]
                alpha_image = scalar_result["alpha_image"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        image = compose_layer(
            mode=mode,
            rgb=rgb,
            unc_image=unc_image,
            alpha_image=alpha_image,
            vmin=0.0 if self.vmin is None else self.vmin,
            vmax=1.0 if self.vmax is None else self.vmax,
            colormap_name=self.colormap,
        )
        jpeg = tensor_to_jpeg(image, quality=quality)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RenderedFrame(
            metadata={
                "mode": mode,
                "width": int(image.shape[2]),
                "height": int(image.shape[1]),
                "elapsed_ms": elapsed_ms,
                "jpeg_quality": quality,
            },
            jpeg=jpeg,
        )
