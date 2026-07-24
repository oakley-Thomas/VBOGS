#!/usr/bin/env python3

"""Render Octree-AnyGS RGB views beside VBOGS uncertainty maps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.octree_config import load_octree_config_for_model
from vbogs.render import render_scalar
from vbogs.dataset_splits import load_split_metadata, metadata_cameras_for_split

DEFAULT_OCTREE_OUTPUT_ROOT = Path("/data/OCTREE-ANYGS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0008_sync",
        help="Drive id used to resolve default model, uncertainty, and output paths.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Octree-AnyGS run directory. Defaults to the latest run under "
            "`/data/OCTREE-ANYGS/<drive>/`."
        ),
    )
    parser.add_argument(
        "--uncertainty",
        type=Path,
        default=None,
        help="Per-anchor uncertainty .npy file. Defaults to `data/m4/<drive>/U.npy`.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to load. `-1` selects the latest available iteration.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "validation", "both", "all"),
        default="both",
        help="Camera split to render.",
    )
    parser.add_argument(
        "--selection-metadata",
        type=Path,
        default=None,
        help=(
            "Prepared NCore metadata.json used for metadata-backed test and "
            "validation cameras. Defaults to /data/COLMAP/<drive>/metadata.json."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=2,
        help=(
            "Octree-AnyGS image divisor/target width for this render pass. "
            "Smaller divisors produce higher-resolution views; defaults to 2, "
            "which renders twice the width and height of a model trained with 4."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root. Defaults to "
            "`outputs/uncertainty_views/<drive>`."
        ),
    )
    parser.add_argument(
        "--colormap",
        default="turbo",
        help="Matplotlib colormap used for uncertainty heatmaps.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum uncertainty value for the fixed heatmap scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum uncertainty value for the fixed heatmap scale.",
    )
    parser.add_argument(
        "--octree-root",
        type=Path,
        default=Path("Octree-AnyGS"),
        help="Path to the Octree-AnyGS submodule.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=0,
        help="Optional smoke-test cap per split. `0` renders all views.",
    )
    parser.add_argument(
        "--ape",
        type=int,
        default=-1,
        help="Appearance embedding code passed through to Octree-AnyGS.",
    )
    parser.add_argument(
        "--load-source-images",
        action="store_true",
        help=(
            "Use the original Octree-AnyGS camera loader, including source image "
            "tensors. By default this command uses metadata-only cameras."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Silence Octree-AnyGS timestamps.")
    return parser.parse_args()


def resolve_model_path(drive: str, model_path: Path | None) -> Path:
    if model_path is not None:
        return model_path.resolve()

    root = DEFAULT_OCTREE_OUTPUT_ROOT / drive
    if not root.exists():
        raise FileNotFoundError(f"No Octree-AnyGS output directory found at {root}")

    candidates = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "config.yaml").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No Octree-AnyGS runs found under {root}")
    return candidates[-1].resolve()


def resolve_uncertainty_path(drive: str, uncertainty: Path | None) -> Path:
    if uncertainty is not None:
        return uncertainty.resolve()
    return (REPO_ROOT / "data" / "m4" / drive / "U.npy").resolve()


def resolve_output_dir(drive: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (REPO_ROOT / "outputs" / "uncertainty_views" / drive).resolve()


def resolve_selection_metadata(drive: str, selection_metadata: Path | None) -> Path:
    if selection_metadata is not None:
        return selection_metadata.resolve()
    return (Path("/data/COLMAP") / drive / "metadata.json").resolve()


def add_octree_to_path(octree_root: Path) -> Path:
    octree_root = octree_root.resolve()
    if not octree_root.exists():
        raise FileNotFoundError(f"Octree-AnyGS root not found: {octree_root}")
    if str(octree_root) not in sys.path:
        sys.path.insert(0, str(octree_root))
    return octree_root


def load_scene(
    model_path: Path,
    octree_root: Path,
    iteration: int,
    ape_code: int,
    quiet: bool,
    resolution: int | None = None,
    load_source_images: bool = False,
):
    add_octree_to_path(octree_root)

    from scene import Scene
    from utils.general_utils import parse_cfg, safe_state

    if not load_source_images:
        from vbogs.viewer.rendering import install_lightweight_octree_camera_loader

        install_lightweight_octree_camera_loader()

    cfg = load_octree_config_for_model(model_path)
    dataset, opt, pipe = parse_cfg(cfg)
    if resolution is not None:
        if resolution == 0:
            raise ValueError("--resolution must be non-zero")
        dataset.resolution = resolution
    dataset.model_path = str(model_path)

    model_config = dataset.model_config
    module_name = "scene." + model_config["kwargs"]["gs_attr"][:-2] + "_model"
    modules = __import__(module_name, fromlist=[""])
    gaussians = getattr(modules, model_config["name"])(**model_config["kwargs"])
    gaussians.ape_code = ape_code

    safe_state(quiet)
    scene = Scene(dataset, opt, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.eval()
    return scene, gaussians, pipe


def load_uncertainty(path: Path, anchor_count: int) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Uncertainty file not found: {path}")
    values = np.load(path)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D uncertainty array, got shape {values.shape}")
    if values.shape[0] != anchor_count:
        raise ValueError(
            f"U.npy has {values.shape[0]} values, but the scene has {anchor_count} anchors"
        )
    return torch.from_numpy(values.astype(np.float32, copy=False)).cuda()


def choose_scale(values: torch.Tensor, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    finite = values.detach().cpu().numpy()
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


def heatmap_tensor(
    values: torch.Tensor,
    alpha: torch.Tensor,
    *,
    vmin: float,
    vmax: float,
    colormap_name: str,
) -> torch.Tensor:
    import matplotlib.cm as cm

    alpha_mask = alpha > 0
    normalized = torch.zeros_like(values)
    normalized[alpha_mask] = (values[alpha_mask] - vmin) / (vmax - vmin)
    normalized = normalized.clamp(0.0, 1.0)

    colormap = cm.get_cmap(colormap_name)
    mapped = colormap(normalized.detach().cpu().numpy())[..., :3]
    heatmap = torch.from_numpy(mapped.astype(np.float32)).permute(2, 0, 1)
    heatmap[:, ~alpha_mask.detach().cpu()] = 0.0
    return heatmap


def image_stem(view: Any, index: int) -> str:
    name = getattr(view, "image_name", "") or f"{index:05d}"
    return Path(str(name)).stem


def split_names_to_render(split: str) -> tuple[str, ...]:
    if split == "both":
        return ("train", "test")
    if split == "all":
        return ("train", "test", "validation")
    return (split,)


def iter_selected_splits(
    scene: Any,
    split: str,
    *,
    split_metadata: dict[str, Any],
    resolution: int | None,
) -> Iterable[tuple[str, list[Any]]]:
    train_cameras = list(scene.getTrainCameras())
    reference = train_cameras[0] if train_cameras else None
    for split_name in split_names_to_render(split):
        if split_name == "train":
            cameras = train_cameras
        elif split_name in ("test", "validation"):
            cameras = metadata_cameras_for_split(
                split_metadata,
                split_name,
                resolution_arg=resolution or 1,
                source_cam=reference,
            )
            if not cameras and split_name == "test":
                cameras = list(scene.getTestCameras())
        else:
            raise ValueError(f"Unsupported split: {split_name}")
        yield split_name, cameras


def ensure_split_dirs(output_dir: Path, split_name: str) -> dict[str, Path]:
    split_root = output_dir / split_name
    paths = {
        "rgb": split_root / "rgb",
        "uncertainty": split_root / "uncertainty",
        "side_by_side": split_root / "side_by_side",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def render_splits(
    scene: Any,
    gaussians: Any,
    pipe: Any,
    uncertainty: torch.Tensor,
    *,
    iteration: int,
    split: str,
    output_dir: Path,
    vmin: float,
    vmax: float,
    colormap_name: str,
    max_views: int,
    split_metadata: dict[str, Any],
    resolution: int | None,
) -> list[dict[str, Any]]:
    from gaussian_renderer.render import render as render_rgb

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for split_name, cameras in iter_selected_splits(
            scene,
            split,
            split_metadata=split_metadata,
            resolution=resolution,
        ):
            out_paths = ensure_split_dirs(output_dir, split_name)
            selected_cameras = cameras[:max_views] if max_views > 0 else cameras
            print(f"Rendering {len(selected_cameras)} {split_name} views")

            for index, view in enumerate(selected_cameras):
                torch.cuda.synchronize()
                start = time.time()

                rgb_pkg = render_rgb(view, gaussians, pipe, scene.background, iteration)
                rgb = torch.clamp(rgb_pkg["render"], 0.0, 1.0)

                unc_image, alpha_image = render_scalar(
                    view,
                    gaussians,
                    pipe,
                    uncertainty,
                    iteration,
                )

                display_unc = torch.zeros_like(unc_image)
                alpha_mask = alpha_image > 0
                display_unc[alpha_mask] = unc_image[alpha_mask] / alpha_image[alpha_mask].clamp_min(
                    1.0e-8
                )
                heatmap = heatmap_tensor(
                    display_unc,
                    alpha_image,
                    vmin=vmin,
                    vmax=vmax,
                    colormap_name=colormap_name,
                ).to(rgb.device)

                stem = image_stem(view, index)
                rgb_path = out_paths["rgb"] / f"{index:05d}_{stem}.png"
                uncertainty_path = out_paths["uncertainty"] / f"{index:05d}_{stem}.png"
                side_by_side_path = out_paths["side_by_side"] / f"{index:05d}_{stem}.png"

                torchvision.utils.save_image(rgb, str(rgb_path))
                torchvision.utils.save_image(heatmap, str(uncertainty_path))
                torchvision.utils.save_image(torch.cat([rgb, heatmap], dim=2), str(side_by_side_path))

                torch.cuda.synchronize()
                elapsed = time.time() - start

                alpha_sum = float(alpha_image.sum().item())
                rows.append(
                    {
                        "split": split_name,
                        "index": index,
                        "image_name": stem,
                        "width": int(view.image_width),
                        "height": int(view.image_height),
                        "alpha_sum": alpha_sum,
                        "uncertainty_mean": (
                            float(unc_image.sum().item() / alpha_sum) if alpha_sum > 0.0 else 0.0
                        ),
                        "rgb": str(rgb_path),
                        "uncertainty": str(uncertainty_path),
                        "side_by_side": str(side_by_side_path),
                        "seconds": elapsed,
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.drive, args.model_path)
    uncertainty_path = resolve_uncertainty_path(args.drive, args.uncertainty)
    output_dir = resolve_output_dir(args.drive, args.output_dir)
    split_metadata_path = resolve_selection_metadata(args.drive, args.selection_metadata)
    split_metadata = load_split_metadata(split_metadata_path)

    print(f"Loading Octree-AnyGS model: {model_path}")
    scene, gaussians, pipe = load_scene(
        model_path,
        args.octree_root,
        args.iteration,
        args.ape,
        args.quiet,
        args.resolution,
        args.load_source_images,
    )
    loaded_iteration = int(scene.loaded_iter)
    anchor_count = int(gaussians.get_anchor.shape[0])

    print(f"Loading uncertainty: {uncertainty_path}")
    uncertainty = load_uncertainty(uncertainty_path, anchor_count)
    vmin, vmax = choose_scale(uncertainty, args.vmin, args.vmax)
    print(f"Using uncertainty color scale: vmin={vmin:.6g}, vmax={vmax:.6g}")

    rows = render_splits(
        scene,
        gaussians,
        pipe,
        uncertainty,
        iteration=loaded_iteration,
        split=args.split,
        output_dir=output_dir,
        vmin=vmin,
        vmax=vmax,
        colormap_name=args.colormap,
        max_views=args.max_views,
        split_metadata=split_metadata,
        resolution=args.resolution,
    )

    metadata = {
        "drive": args.drive,
        "model_path": str(model_path),
        "uncertainty_path": str(uncertainty_path),
        "iteration": loaded_iteration,
        "split": args.split,
        "resolution": args.resolution,
        "vmin": vmin,
        "vmax": vmax,
        "colormap": args.colormap,
        "selection_metadata": str(split_metadata_path) if split_metadata else None,
        "anchor_count": anchor_count,
        "view_count": len(rows),
        "views": rows,
    }
    save_json(output_dir / "metadata.json", metadata)
    print(f"Wrote {len(rows)} side-by-side renders under {output_dir}")


if __name__ == "__main__":
    main()
