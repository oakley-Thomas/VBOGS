#!/usr/bin/env python3

"""Generate a 16 GB-safe Octree-AnyGS config and optionally launch training."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


LOCAL_16GB_CONFIG: Dict[str, Any] = {
    "model_params": {
        "model_config": {
            "name": "GaussianLoDModel",
            "kwargs": {
                "gs_attr": "implicit3D",
                "color_attr": "RGB",
                "feat_dim": 16,
                "view_dim": 3,
                "appearance_dim": 0,
                "n_offsets": 10,
                "visible_threshold": 0.02,
                "base_layer": 9,
                "dist_ratio": 0.995,
                "render_mode": "RGB",
            },
        },
        "dataset_name": "kitti360",
        "scene_name": "unset_scene",
        "resolution": 4,
        "white_background": False,
        "random_background": False,
        "resolution_scales": [1.0],
        "data_device": "cpu",
        "eval": True,
        "ratio": 1,
        "data_format": "colmap",
        "llffhold": 8,
        "add_mask": False,
        "add_depth": False,
    },
    "pipeline_params": {
        "vis_step": 2500,
        "add_prefilter": False,
    },
    "optim_params": {
        "iterations": 15000,
        "position_lr_init": 0.0,
        "position_lr_final": 0.0,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 15000,
        "offset_lr_init": 0.001,
        "offset_lr_final": 0.00001,
        "offset_lr_delay_mult": 0.01,
        "offset_lr_max_steps": 15000,
        "feature_lr": 0.0075,
        "scaling_lr": 0.007,
        "rotation_lr": 0.002,
        "mlp_opacity_lr_init": 0.002,
        "mlp_opacity_lr_final": 0.00002,
        "mlp_opacity_lr_delay_mult": 0.01,
        "mlp_opacity_lr_max_steps": 15000,
        "mlp_cov_lr_init": 0.004,
        "mlp_cov_lr_final": 0.004,
        "mlp_cov_lr_delay_mult": 0.01,
        "mlp_cov_lr_max_steps": 15000,
        "mlp_color_lr_init": 0.008,
        "mlp_color_lr_final": 0.00005,
        "mlp_color_lr_delay_mult": 0.01,
        "mlp_color_lr_max_steps": 15000,
        "appearance_lr_init": 0.05,
        "appearance_lr_final": 0.0005,
        "appearance_lr_delay_mult": 0.01,
        "appearance_lr_max_steps": 15000,
        "lambda_dssim": 0.2,
        "lambda_dreg": 0.01,
        "lambda_normal": 0.0,
        "normal_start_iter": 7000,
        "lambda_dist": 0.0,
        "dist_start_iter": 3000,
        "start_depth": 500,
        "depth_l1_weight_init": 1.0,
        "depth_l1_weight_final": 0.01,
        "progressive": True,
        "coarse_iter": 4000,
        "coarse_factor": 1.5,
        "start_stat": 20000,
        "update_from": 20000,
        "update_interval": 200,
        "update_until": 0,
        "min_opacity": 0.005,
        "success_threshold": 0.8,
        "densify_grad_threshold": 0.0002,
        "update_ratio": 0.15,
        "extra_ratio": 0.15,
        "extra_up": 0.01,
        "overlap": False,
        "densification": False,
        "growing_strategy": "mean",
    },
}

DEFAULT_OUTPUT_ROOT = Path("/data/OCTREE-ANYGS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Prepared Octree-AnyGS dataset directory with images/ and sparse/0/.",
    )
    parser.add_argument(
        "--scene-name",
        default="",
        help="Override scene name stored in the generated config.",
    )
    parser.add_argument(
        "--dataset-name",
        default="kitti360",
        help=(
            "Logical dataset name used by VBOGS. Octree-AnyGS output placement is "
            "controlled by --output-root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for Octree-AnyGS training outputs. Runs are written to "
            "`<output-root>/<scene-name>/<timestamp>/`."
        ),
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Optional explicit path for the generated YAML config.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=4,
        help="Octree-AnyGS image divisor. 4 is the conservative 16 GB default.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15000,
        help="Training iterations for the generated config.",
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="Held-out test frame cadence for eval mode.",
    )
    parser.add_argument(
        "--feat-dim",
        type=int,
        default=16,
        help="Anchor feature dimension. Lower values reduce VRAM pressure.",
    )
    parser.add_argument(
        "--base-layer",
        type=int,
        default=9,
        help="LoD base layer. Lower values reduce anchor count and memory.",
    )
    parser.add_argument(
        "--visible-threshold",
        type=float,
        default=0.02,
        help="LoD pruning visibility threshold.",
    )
    parser.add_argument(
        "--gpu",
        default="-1",
        help="GPU id passed through to Octree-AnyGS/train.py.",
    )
    parser.add_argument(
        "--write-config-only",
        action="store_true",
        help="Generate the config but do not launch training.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch Octree-AnyGS/train.py.",
    )
    parser.add_argument(
        "--octree-root",
        type=Path,
        default=Path("Octree-AnyGS"),
        help="Path to the Octree-AnyGS submodule.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(LOCAL_16GB_CONFIG)
    scene_name = args.scene_name or args.dataset_path.name
    model_params = cfg["model_params"]
    model_params["source_path"] = str(args.dataset_path.resolve())
    model_params["scene_name"] = scene_name
    # Upstream Octree-AnyGS hardcodes outputs as:
    #   outputs/<dataset_name>/<scene_name>/<timestamp>
    # Supplying an absolute path here makes os.path.join ignore the leading
    # "outputs" segment, giving VBOGS a repo-owned output root without editing
    # the read-only submodule.
    model_params["dataset_name"] = str(args.output_root.resolve())
    model_params["vbogs_dataset_name"] = args.dataset_name
    model_params["resolution"] = args.resolution
    model_params["llffhold"] = args.llffhold

    model_kwargs = model_params["model_config"]["kwargs"]
    model_kwargs["feat_dim"] = args.feat_dim
    model_kwargs["base_layer"] = args.base_layer
    model_kwargs["visible_threshold"] = args.visible_threshold

    optim_params = cfg["optim_params"]
    optim_params["iterations"] = args.iterations
    optim_params["position_lr_max_steps"] = args.iterations
    optim_params["offset_lr_max_steps"] = args.iterations
    optim_params["mlp_opacity_lr_max_steps"] = args.iterations
    optim_params["mlp_cov_lr_max_steps"] = args.iterations
    optim_params["mlp_color_lr_max_steps"] = args.iterations
    optim_params["appearance_lr_max_steps"] = args.iterations
    optim_params["update_until"] = min(args.iterations - 1000, optim_params["update_until"])
    return cfg


def resolve_output_config(args: argparse.Namespace) -> Path:
    if args.output_config is not None:
        return args.output_config
    scene_name = args.scene_name or args.dataset_path.name
    return Path("generated_configs") / f"{scene_name}_octree_anygs_16gb.yaml"


def write_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    config_path = resolve_output_config(args)
    write_config(config_path, cfg)
    print(f"Wrote config: {config_path}")

    if args.write_config_only:
        return

    octree_root = args.octree_root.resolve()
    train_script = octree_root / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Octree-AnyGS training script not found: {train_script}")

    cmd = [
        args.python,
        str(train_script),
        "--config",
        str(config_path.resolve()),
        "--gpu",
        str(args.gpu),
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(octree_root.parent), check=True)


if __name__ == "__main__":
    main()
