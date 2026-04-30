#!/usr/bin/env python3

"""Score next-best-view candidates by rendering per-anchor uncertainty.

This is the first M6 entry point. It loads a trained Octree-AnyGS scene plus
`U.npy`, renders scalar uncertainty for candidate cameras, and writes top-K
diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json
from vbogs.render import render_scalar

DEFAULT_OCTREE_OUTPUT_ROOT = Path("/data/OCTREE-ANYGS")


class ScalarCam:
    """Small camera object compatible with Octree-AnyGS `generate_gaussians`."""

    def __init__(self, source_cam, c2w: np.ndarray, uid: int, image_name: str):
        self.uid = uid
        self.image_name = image_name
        self.image_path = getattr(source_cam, "image_path", "")
        self.resolution_scale = float(source_cam.resolution_scale)
        self.image_width = int(source_cam.image_width)
        self.image_height = int(source_cam.image_height)
        self.FoVx = float(source_cam.FoVx)
        self.FoVy = float(source_cam.FoVy)
        self.fx = float(source_cam.fx)
        self.fy = float(source_cam.fy)
        self.cx = float(source_cam.cx)
        self.cy = float(source_cam.cy)
        self.znear = float(getattr(source_cam, "znear", 0.01))
        self.zfar = float(getattr(source_cam, "zfar", 100.0))
        self.c2w_np = np.asarray(c2w, dtype=np.float32)

        world_to_view = np.linalg.inv(self.c2w_np)
        self.world_view_transform = torch.tensor(world_to_view, dtype=torch.float32, device="cuda").transpose(0, 1)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default="2013_05_28_drive_0008_sync")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Octree-AnyGS run directory. Defaults to latest `/data/OCTREE-ANYGS/<drive>` run.",
    )
    parser.add_argument(
        "--u-path",
        type=Path,
        default=None,
        help="Path to M5 `U.npy`. Defaults to `data/m4/<drive>/U.npy`.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--octree-root", type=Path, default=Path("Octree-AnyGS"))
    parser.add_argument("--candidate-source", choices=("test", "train", "lattice"), default="test")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--force-all-levels", action="store_true")
    parser.add_argument("--save-top-images", type=int, default=0)
    parser.add_argument("--reference-source", choices=("test", "train"), default="test")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--lattice-radii", default="0,5,10")
    parser.add_argument("--lattice-yaws-deg", default="-30,0,30")
    parser.add_argument("--lattice-plane", choices=("xy", "xz"), default="xy")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path.resolve()
    root = DEFAULT_OCTREE_OUTPUT_ROOT / args.drive
    candidates = sorted(path for path in root.glob("*") if path.is_dir() and (path / "config.yaml").exists())
    if not candidates:
        raise FileNotFoundError(f"No Octree-AnyGS runs found under {root}")
    return candidates[-1].resolve()


def resolve_u_path(args: argparse.Namespace) -> Path:
    if args.u_path is not None:
        return args.u_path.resolve()
    return (Path("data/m4") / args.drive / "U.npy").resolve()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def add_octree_to_path(octree_root: Path) -> None:
    octree_root = octree_root.resolve()
    if str(octree_root) not in sys.path:
        sys.path.insert(0, str(octree_root))


def load_octree_scene(model_path: Path, iteration: int, octree_root: Path, quiet: bool):
    add_octree_to_path(octree_root)
    from scene import Scene
    from utils.general_utils import parse_cfg, safe_state

    with (model_path / "config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    dataset, opt, pipe = parse_cfg(cfg)
    dataset.model_path = str(model_path)

    safe_state(quiet)
    model_config = dataset.model_config
    modules = __import__(f"scene.{model_config['kwargs']['gs_attr'][:-2]}_model", fromlist=[""])
    gaussians = getattr(modules, model_config["name"])(**model_config["kwargs"])
    gaussians.ape_code = -1
    scene = Scene(dataset, opt, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.eval()
    return scene, gaussians, pipe


def camera_to_c2w(cam) -> np.ndarray:
    return cam.world_view_transform.transpose(0, 1).inverse().detach().cpu().numpy().astype(np.float32)


def yaw_matrix(angle_rad: float, axis: int) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    mat = np.eye(3, dtype=np.float32)
    axes = [0, 1, 2]
    axes.remove(axis)
    a, b = axes
    mat[a, a] = c
    mat[a, b] = -s
    mat[b, a] = s
    mat[b, b] = c
    return mat


def make_lattice_candidates(reference_cam, args: argparse.Namespace) -> List[ScalarCam]:
    radii = parse_float_list(args.lattice_radii)
    yaws = parse_float_list(args.lattice_yaws_deg)
    ref_c2w = camera_to_c2w(reference_cam)
    plane_axes = (0, 1) if args.lattice_plane == "xy" else (0, 2)
    up_axis = 2 if args.lattice_plane == "xy" else 1

    candidates: List[ScalarCam] = []
    uid = 0
    for radius in radii:
        directions = [0.0] if radius == 0 else [0.0, 90.0, 180.0, 270.0]
        for direction_deg in directions:
            direction_rad = math.radians(direction_deg)
            offset = np.zeros((3,), dtype=np.float32)
            offset[plane_axes[0]] = radius * math.cos(direction_rad)
            offset[plane_axes[1]] = radius * math.sin(direction_rad)
            for yaw_deg in yaws:
                c2w = np.array(ref_c2w, copy=True)
                c2w[:3, 3] = ref_c2w[:3, 3] + offset
                c2w[:3, :3] = yaw_matrix(math.radians(yaw_deg), up_axis) @ ref_c2w[:3, :3]
                candidates.append(ScalarCam(reference_cam, c2w, uid, f"lattice_{uid:04d}"))
                uid += 1
    return candidates


def select_base_cameras(scene, source: str) -> Sequence:
    return scene.getTestCameras() if source == "test" else scene.getTrainCameras()


def build_candidates(scene, args: argparse.Namespace) -> List:
    if args.candidate_source in ("test", "train"):
        candidates = list(select_base_cameras(scene, args.candidate_source))
    else:
        reference_cams = list(select_base_cameras(scene, args.reference_source))
        if not reference_cams:
            raise ValueError(f"No {args.reference_source} cameras available for lattice reference")
        reference = reference_cams[min(args.reference_index, len(reference_cams) - 1)]
        candidates = make_lattice_candidates(reference, args)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("No NBV candidates selected")
    return candidates


def score_candidate(cam, pc, per_anchor_scalar: torch.Tensor, args: argparse.Namespace) -> Dict:
    rendered = render_scalar(
        cam,
        pc,
        per_anchor_scalar,
        iteration=2_147_483_647 if args.iteration == -1 else args.iteration,
        force_all_levels=args.force_all_levels,
    )
    unc_image = rendered["unc_image"]
    alpha_image = rendered["alpha_image"]
    unc_sum = float(unc_image.sum().detach().cpu())
    alpha_sum = float(alpha_image.sum().detach().cpu())
    score = unc_sum / (alpha_sum + args.eps)
    return {
        "score": score,
        "unc_sum": unc_sum,
        "alpha_sum": alpha_sum,
        "visible_anchor_count": int(rendered["visible_mask"].sum().detach().cpu()),
        "visible_gaussian_count": int(rendered["visibility_filter"].sum().detach().cpu()),
        "unc_image": unc_image.detach().cpu().numpy().astype(np.float32),
        "alpha_image": alpha_image.detach().cpu().numpy().astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args)
    u_path = resolve_u_path(args)
    output_root = (args.output_root or (Path("data/m6") / args.drive)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scene, gaussians, _pipe = load_octree_scene(model_path, args.iteration, args.octree_root, args.quiet)
    u_values_np = np.load(u_path).astype(np.float32)
    if u_values_np.shape[0] != gaussians.get_anchor.shape[0]:
        raise ValueError(
            f"U length {u_values_np.shape[0]} does not match anchor count {gaussians.get_anchor.shape[0]}"
        )
    per_anchor_scalar = torch.from_numpy(u_values_np).to(device="cuda", dtype=torch.float32)
    candidates = build_candidates(scene, args)

    rows = []
    top_payloads = []
    with torch.no_grad():
        for idx, cam in enumerate(candidates):
            result = score_candidate(cam, gaussians, per_anchor_scalar, args)
            row = {
                "rank": None,
                "candidate_index": idx,
                "image_name": getattr(cam, "image_name", f"candidate_{idx:04d}"),
                "score": result["score"],
                "unc_sum": result["unc_sum"],
                "alpha_sum": result["alpha_sum"],
                "visible_anchor_count": result["visible_anchor_count"],
                "visible_gaussian_count": result["visible_gaussian_count"],
                "camera_center": [float(x) for x in cam.camera_center.detach().cpu().tolist()],
            }
            rows.append(row)
            top_payloads.append((row, result["unc_image"], result["alpha_image"]))
            if not args.quiet:
                print(
                    f"[{idx + 1:>4}/{len(candidates)}] score={row['score']:.6g} "
                    f"alpha={row['alpha_sum']:.3f} name={row['image_name']}"
                )

    order = sorted(range(len(rows)), key=lambda idx: rows[idx]["score"], reverse=True)
    ranked = []
    for rank, idx in enumerate(order, start=1):
        row = dict(rows[idx])
        row["rank"] = rank
        ranked.append(row)

    top_k = ranked[: max(args.top_k, 1)]
    diagnostics = {
        "drive": args.drive,
        "model_path": str(model_path),
        "u_path": str(u_path),
        "candidate_source": args.candidate_source,
        "candidate_count": len(candidates),
        "top_k": top_k,
        "best_pose": top_k[0],
    }
    save_json(output_root / "nbv_scores.json", diagnostics)

    if args.save_top_images > 0:
        image_root = output_root / "top_images"
        image_root.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(top_k[: args.save_top_images], start=1):
            payload = top_payloads[row["candidate_index"]]
            np.save(image_root / f"rank_{rank:02d}_unc.npy", payload[1])
            np.save(image_root / f"rank_{rank:02d}_alpha.npy", payload[2])

    print(f"Wrote {output_root / 'nbv_scores.json'}")
    print(
        f"Best candidate: index={top_k[0]['candidate_index']} "
        f"score={top_k[0]['score']:.6g} image={top_k[0]['image_name']}"
    )


if __name__ == "__main__":
    main()
