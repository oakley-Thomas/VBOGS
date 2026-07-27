#!/usr/bin/env python3

"""Evaluate held-out Octree RGB renders and optional VBOGS uncertainty."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_uncertainty_views import (
    add_octree_to_path,
    choose_scale,
    heatmap_tensor,
    load_scene,
    load_uncertainty,
)
from vbogs.dataset_splits import load_split_metadata, metadata_cameras_for_split
from vbogs.dynamic_masking import load_manifest, mask_path
from vbogs.uncertainty_evaluation import (
    directory_sha256,
    evaluation_summary,
    evaluation_views,
    file_sha256,
    split_fingerprint,
    view_score_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", choices=("kitti360", "nvidia_ncore"), required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--selection-metadata", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--u-path", type=Path, default=None)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument(
        "--image-mask-root",
        type=Path,
        default=None,
        help=(
            "Dynamic-mask artifact root used for static-region metrics. This is "
            "distinct from --mask-path, which remains the per-anchor observed mask."
        ),
    )
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ape", type=int, default=-1)
    parser.add_argument("--octree-root", type=Path, default=Path("Octree-AnyGS"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def load_ground_truth(path: Path, width: int, height: int, device: str):
    from utils.general_utils import PILtoTorch

    if not path.exists():
        raise FileNotFoundError(f"Prepared held-out image not found: {path}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return PILtoTorch(rgb, (width, height))[:3].to(device).clamp(0.0, 1.0)


def metric_values(render, ground_truth, lpips_model) -> dict[str, float]:
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    # Octree-AnyGS's PSNR helper flattens with Tensor.view(), which requires a
    # contiguous layout. PILtoTorch returns a channel-permuted tensor and model
    # renders may also be sliced views, so normalize both layouts here.
    render_batch = render.contiguous().unsqueeze(0)
    gt_batch = ground_truth.contiguous().unsqueeze(0)
    return {
        "PSNR": float(psnr(render_batch, gt_batch).mean().item()),
        "SSIM": float(ssim(render_batch, gt_batch).mean().item()),
        # Match the upstream Octree evaluator: LPIPS receives clamped [0, 1] RGB.
        "LPIPS": float(lpips_model(render_batch, gt_batch).mean().item()),
    }


def load_static_mask_for_evaluation(mask_root: Path, image_name: str, source_shape: tuple[int, int], target_shape: tuple[int, int]):
    """Load a strict binary source-resolution mask and resize for rendering."""

    path = mask_path(mask_root, image_name)
    if not path.is_file():
        raise FileNotFoundError(f"Dynamic mask missing for {image_name}: {path}")
    with Image.open(path) as image:
        raw = np.asarray(image.convert("L"))
    if raw.shape != source_shape:
        raise ValueError(f"Dynamic mask {path} has shape {raw.shape}, expected {source_shape}")
    values = set(np.unique(raw).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"Dynamic mask {path} must be binary 0/255, found {sorted(values)}")
    if raw.shape != target_shape:
        raw = np.asarray(
            Image.fromarray(raw).resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
        )
    return raw == 255


def static_metric_values(render, ground_truth, static_mask, lpips_model) -> dict[str, float | int]:
    """Metrics restricted to confirmed static pixels.

    SSIM uses only output locations whose full 11x11 receptive field is static.
    LPIPS is evaluated over deterministic non-overlapping all-static 64px tiles.
    """

    import torch
    import torch.nn.functional as F
    from utils.loss_utils import create_window

    if static_mask.ndim != 2 or tuple(static_mask.shape) != tuple(render.shape[-2:]):
        raise ValueError("Static mask must match rendered image height and width")
    mask = static_mask.to(device=render.device, dtype=render.dtype)
    static_pixels = int(mask.sum().item())
    if static_pixels == 0:
        raise ValueError("Static mask contains no static pixels")
    error = (render - ground_truth).pow(2)
    mse = (error * mask.unsqueeze(0)).sum() / (mask.sum() * render.shape[0])
    psnr = float("inf") if float(mse) == 0.0 else float(-10.0 * torch.log10(mse).item())

    render_batch = render.contiguous().unsqueeze(0)
    gt_batch = ground_truth.contiguous().unsqueeze(0)
    window = create_window(11, int(render.shape[0])).to(device=render.device, dtype=render.dtype)
    # Recreate the upstream SSIM map so a complete static support can be selected.
    mu1 = F.conv2d(render_batch, window, padding=5, groups=render.shape[0])
    mu2 = F.conv2d(gt_batch, window, padding=5, groups=render.shape[0])
    sigma1 = F.conv2d(render_batch * render_batch, window, padding=5, groups=render.shape[0]) - mu1.pow(2)
    sigma2 = F.conv2d(gt_batch * gt_batch, window, padding=5, groups=render.shape[0]) - mu2.pow(2)
    sigma12 = F.conv2d(render_batch * gt_batch, window, padding=5, groups=render.shape[0]) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + 0.01**2) * (2 * sigma12 + 0.03**2)) / ((mu1.pow(2) + mu2.pow(2) + 0.01**2) * (sigma1 + sigma2 + 0.03**2))
    support = F.conv2d(mask[None, None], torch.ones((1, 1, 11, 11), device=render.device, dtype=render.dtype), padding=5)
    valid_ssim = support[0, 0] == 121
    ssim_static = float("nan")
    if bool(valid_ssim.any()):
        ssim_static = float(ssim_map[0, :, valid_ssim].mean().item())

    tiles_render = []
    tiles_gt = []
    tile_size = 64
    height, width = mask.shape
    for top in range(0, height - tile_size + 1, tile_size):
        for left in range(0, width - tile_size + 1, tile_size):
            tile_mask = mask[top : top + tile_size, left : left + tile_size]
            if bool(torch.all(tile_mask > 0)):
                tiles_render.append(render[:, top : top + tile_size, left : left + tile_size])
                tiles_gt.append(ground_truth[:, top : top + tile_size, left : left + tile_size])
    lpips_static = float("nan")
    if tiles_render:
        lpips_static = float(lpips_model(torch.stack(tiles_render), torch.stack(tiles_gt)).mean().item())
    return {
        "PSNR_static": psnr,
        "SSIM_static": ssim_static,
        "LPIPS_static": lpips_static,
        "static_pixel_fraction": float(mask.mean().item()),
        "static_lpips_tile_count": len(tiles_render),
    }


def image_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        "render": output_dir / "renders" / "rgb",
        "ground_truth": output_dir / "renders" / "ground_truth",
        "error": output_dir / "renders" / "absolute_error",
        "uncertainty": output_dir / "renders" / "uncertainty",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    if args.resolution == 0:
        raise ValueError("--resolution must be non-zero")
    if args.max_views < 0:
        raise ValueError("--max-views cannot be negative")

    metadata_path = args.selection_metadata.resolve()
    metadata = load_split_metadata(metadata_path)
    metadata_dataset = metadata.get("dataset", "kitti360")
    if metadata_dataset != args.dataset_name:
        raise ValueError(
            f"Metadata dataset {metadata_dataset!r} does not match --dataset-name {args.dataset_name!r}"
        )
    descriptors = evaluation_views(metadata, args.split)

    model_path = args.model_path.resolve()
    scene, gaussians, pipe = load_scene(
        model_path,
        args.octree_root,
        args.iteration,
        args.ape,
        args.quiet,
        args.resolution,
        False,
    )
    loaded_iteration = int(scene.loaded_iter)
    if loaded_iteration != args.iteration:
        raise ValueError(
            f"Requested iteration {args.iteration}, but Octree loaded {loaded_iteration}"
        )
    train_cameras = list(scene.getTrainCameras())
    reference = train_cameras[0] if train_cameras else None
    cameras = metadata_cameras_for_split(
        metadata,
        args.split,
        resolution_arg=args.resolution,
        source_cam=reference,
    )
    if len(cameras) != len(descriptors):
        raise ValueError(
            f"Camera/metadata mismatch: {len(cameras)} render cameras for {len(descriptors)} records"
        )
    if args.max_views:
        cameras = cameras[: args.max_views]
        descriptors = descriptors[: args.max_views]

    add_octree_to_path(args.octree_root)
    import lpips
    import torch
    import torchvision
    from gaussian_renderer.render import render as render_rgb

    lpips_model = lpips.LPIPS(net="vgg").cuda().eval()
    uncertainty = None
    uncertainty_path = None
    observed_mask = None
    observed_mask_path = None
    vmin = vmax = None
    if args.u_path is not None:
        uncertainty_path = args.u_path.resolve()
        uncertainty = load_uncertainty(uncertainty_path, int(gaussians.get_anchor.shape[0]))
        vmin, vmax = choose_scale(uncertainty, None, None)
        candidate_mask_path = args.mask_path
        if candidate_mask_path is None:
            sibling = uncertainty_path.with_name("observed_mask.npy")
            candidate_mask_path = sibling if sibling.exists() else None
        if candidate_mask_path is not None:
            observed_mask_path = candidate_mask_path.resolve()
            observed_mask = load_uncertainty(
                observed_mask_path, int(gaussians.get_anchor.shape[0])
            ).to(dtype=uncertainty.dtype)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = image_paths(output_dir) if args.save_images else None
    prepared_images = metadata_path.parent / "images"
    image_mask_root = args.image_mask_root.resolve() if args.image_mask_root is not None else None
    image_mask_manifest = None
    if image_mask_root is not None:
        image_mask_manifest = load_manifest(image_mask_root)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, (camera, descriptor) in enumerate(zip(cameras, descriptors)):
            camera_name = Path(str(camera.image_name)).as_posix()
            expected_stem = Path(descriptor.image_name).stem
            if Path(camera_name).stem != expected_stem:
                raise ValueError(
                    f"Camera order mismatch at {index}: {camera_name!r} vs {descriptor.image_name!r}"
                )

            render_package = render_rgb(camera, gaussians, pipe, scene.background, loaded_iteration)
            rendered = render_package["render"][:3].clamp(0.0, 1.0)
            ground_truth = load_ground_truth(
                prepared_images / descriptor.image_name,
                int(camera.image_width),
                int(camera.image_height),
                str(rendered.device),
            )
            if rendered.shape != ground_truth.shape:
                raise ValueError(
                    f"Render/ground-truth shape mismatch for {descriptor.view_id}: "
                    f"{tuple(rendered.shape)} vs {tuple(ground_truth.shape)}"
                )

            values = metric_values(rendered, ground_truth, lpips_model)
            static_mask = None
            static_values: dict[str, float | int] = {}
            if image_mask_root is not None:
                with Image.open(prepared_images / descriptor.image_name) as source_image:
                    source_shape = (source_image.height, source_image.width)
                static_mask_np = load_static_mask_for_evaluation(
                    image_mask_root,
                    descriptor.image_name,
                    source_shape,
                    (int(rendered.shape[1]), int(rendered.shape[2])),
                )
                static_mask = torch.from_numpy(static_mask_np)
                static_values = static_metric_values(rendered, ground_truth, static_mask, lpips_model)
            row: dict[str, Any] = {
                **descriptor.to_json(),
                "index": index,
                "width": int(camera.image_width),
                "height": int(camera.image_height),
                **values,
                "alpha_sum": None,
                "uncertainty_sum": None,
                "uncertainty_score": None,
                "observed_sum": None,
                "observed_uncertainty_sum": None,
                "uncertainty_score_observed": None,
                "unobserved_fraction": None,
                "uncertainty_score_static": None,
                **static_values,
            }

            heatmap = None
            if uncertainty is not None:
                from vbogs.render import render_scalar

                unc_image, alpha_image = render_scalar(
                    camera,
                    gaussians,
                    pipe,
                    uncertainty,
                    loaded_iteration,
                )
                alpha_sum = float(alpha_image.sum().item())
                uncertainty_sum = float(unc_image.sum().item())
                row.update(
                    {
                        "alpha_sum": alpha_sum,
                        "uncertainty_sum": uncertainty_sum,
                        "uncertainty_score": uncertainty_sum / (alpha_sum + 1.0e-8),
                    }
                )
                if static_mask is not None:
                    static_mask_device = static_mask.to(device=unc_image.device, dtype=unc_image.dtype)
                    static_alpha_sum = float((alpha_image * static_mask_device).sum().item())
                    static_uncertainty_sum = float((unc_image * static_mask_device).sum().item())
                    row.update(
                        {
                            "static_alpha_sum": static_alpha_sum,
                            "static_uncertainty_sum": static_uncertainty_sum,
                            "uncertainty_score_static": static_uncertainty_sum / (static_alpha_sum + 1.0e-8),
                        }
                    )
                if observed_mask is not None:
                    observed_image, _ = render_scalar(
                        camera, gaussians, pipe, observed_mask, loaded_iteration
                    )
                    observed_unc_image, _ = render_scalar(
                        camera, gaussians, pipe, uncertainty * observed_mask, loaded_iteration
                    )
                    row.update(
                        view_score_fields(
                            uncertainty_sum,
                            float(observed_unc_image.sum().item()),
                            float(observed_image.sum().item()),
                            alpha_sum,
                        )
                    )
                if saved_paths is not None:
                    display_uncertainty = torch.zeros_like(unc_image)
                    visible = alpha_image > 0
                    display_uncertainty[visible] = unc_image[visible] / alpha_image[visible].clamp_min(1.0e-8)
                    heatmap = heatmap_tensor(
                        display_uncertainty,
                        alpha_image,
                        vmin=float(vmin),
                        vmax=float(vmax),
                        colormap_name="turbo",
                    ).to(rendered.device)

            if saved_paths is not None:
                safe_name = str(Path(descriptor.view_id).with_suffix("")).replace("/", "__")
                filename = f"{index:05d}_{safe_name}.png"
                torchvision.utils.save_image(rendered, saved_paths["render"] / filename)
                torchvision.utils.save_image(ground_truth, saved_paths["ground_truth"] / filename)
                torchvision.utils.save_image((rendered - ground_truth).abs(), saved_paths["error"] / filename)
                if heatmap is not None:
                    torchvision.utils.save_image(heatmap, saved_paths["uncertainty"] / filename)
            rows.append(row)

    summary = evaluation_summary(rows)
    provenance = {
        "schema_version": 2,
        "dataset": args.dataset_name,
        "scene_id": args.scene_id,
        "split": args.split,
        "model_path": str(model_path),
        "iteration": loaded_iteration,
        "resolution": args.resolution,
        "selection_metadata": str(metadata_path),
        "split_hash": split_fingerprint(metadata),
        "uncertainty_path": str(uncertainty_path) if uncertainty_path else None,
        "uncertainty_sha256": file_sha256(uncertainty_path) if uncertainty_path else None,
        "mask_path": str(observed_mask_path) if observed_mask_path else None,
        "mask_sha256": file_sha256(observed_mask_path) if observed_mask_path else None,
        "image_mask_root": str(image_mask_root) if image_mask_root else None,
        "image_mask_sha256": directory_sha256(image_mask_root) if image_mask_root else None,
        "image_mask_manifest": image_mask_manifest,
        "view_count": len(rows),
    }
    save_json(output_dir / "per_view.json", {**provenance, "views": rows})
    save_json(output_dir / "summary.json", {**provenance, **summary})
    print(f"Wrote held-out evaluation for {len(rows)} {args.split} views to {output_dir}")


if __name__ == "__main__":
    main()
