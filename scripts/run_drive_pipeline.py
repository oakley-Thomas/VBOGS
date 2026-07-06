#!/usr/bin/env python3

"""Run the implemented VBOGS pipeline for one scene.

This is the internal implementation behind ``scripts/run_pipeline.sh``. Run the
operator entrypoint from inside ``vbogs-pipeline`` so it can resolve sibling
containers and call ``docker exec`` through the mounted Docker socket.
The framework boundary stays explicit:

- `vbogs-torch` runs dataset prep, Octree-AnyGS training, stereo export, and
  point-to-anchor bucketing.
- `vbogs-jax` runs per-anchor VBGS fitting and fit inspection.

Data moves only through the shared stack mounts available at the same paths in
both containers.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


STAGES = (
    "prepare",
    "train",
    "stereo",
    "bucket",
    "fit",
    "inspect",
    "uncertainty",
    "map-viz",
    "render",
    "nbv",
    "nbv-viz",
    "bundle",
)
TORCH_SERVICE = "vbogs-torch"
JAX_SERVICE = "vbogs-jax"
DEFAULT_CONFIG = Path("configs/pipeline/default.yaml")
CONFIG_KEY_MAP = {
    "dataset": {
        "name": "dataset_name",
        "scene_id": "scene_id",
        "ncore_root": "ncore_root",
        "camera_ids": "camera_ids",
        "point_source": "point_source",
        "camera_depth_pair": "camera_depth_pair",
        "lidar_id": "ncore_lidar_id",
    },
    "pipeline": {
        "drive": "drive",
        "start_at": "start_at",
        "stop_after": "stop_after",
        "dry_run": "dry_run",
    },
    "inputs": {
        "raw_root": "raw_root",
        "poses_root": "poses_root",
        "calibration_dir": "calibration_dir",
    },
    "prepare": {
        "frame_step": "frame_step",
        "max_frames": "max_frames",
        "copy_mode": "copy_mode",
        "training_cameras": "training_cameras",
        "seed_mode": "seed_mode",
    },
    "train": {
        "gpu": "gpu",
        "resolution": "resolution",
        "iterations": "iterations",
        "llffhold": "llffhold",
        "gaussian_type": "gaussian_type",
        "feat_dim": "feat_dim",
        "base_layer": "base_layer",
        "visible_threshold": "visible_threshold",
        "port": "train_port",
        "write_config_only": "write_config_only",
        "skip_stack_check": "skip_stack_check",
    },
    "stereo": {
        "matcher": "matcher",
        "pixel_step": "pixel_step",
        "max_points_per_frame": "max_points_per_frame",
        "write_ply": "write_ply",
    },
    "points": {
        "point_source": "point_source",
        "matcher": "matcher",
        "pixel_step": "pixel_step",
        "max_points_per_frame": "max_points_per_frame",
        "write_ply": "write_ply",
    },
    "bucket": {
        "model_path": "model_path",
        "bucket_iteration": "bucket_iteration",
        "point_chunk_size": "bucket_point_chunk_size",
        "max_points": "bucket_max_points",
    },
    "fit": {
        "jax_device": "jax_device",
        "fit_mode": "fit_mode",
        "batch_size": "batch_size",
        "batch_buckets": "batch_buckets",
        "no_auto_extend_buckets": "no_auto_extend_buckets",
        "vmap_group_size": "vmap_group_size",
        "max_padded_points_per_group": "max_padded_points_per_group",
        "log_every": "log_every",
    },
    "inspect": {
        "top_k": "inspect_top_k",
        "sample_points": "inspect_sample_points",
        "anchor_id": "inspect_anchor_id",
        "export_ply": "inspect_export_ply",
    },
    "uncertainty": {
        "u_max": "uncertainty_u_max",
        "no_histogram": "uncertainty_no_histogram",
    },
    "map_viz": {
        "output_dir": "map_viz_output_dir",
        "vmin": "map_viz_vmin",
        "vmax": "map_viz_vmax",
        "percentile_low": "map_viz_percentile_low",
        "percentile_high": "map_viz_percentile_high",
        "observed_only": "map_viz_observed_only",
        "no_split_levels": "map_viz_no_split_levels",
        "no_trajectory": "map_viz_no_trajectory",
    },
    "render": {
        "split": "render_split",
        "resolution": "render_resolution",
        "max_views": "render_max_views",
        "colormap": "render_colormap",
        "vmin": "render_vmin",
        "vmax": "render_vmax",
        "output_dir": "render_output_dir",
    },
    "nbv": {
        "candidate_source": "nbv_candidate_source",
        "max_candidates": "nbv_max_candidates",
        "top_k": "nbv_top_k",
        "save_top_images": "nbv_save_top_images",
        "force_all_levels": "nbv_force_all_levels",
        "output_dir": "nbv_output_dir",
    },
    "outputs": {
        "run_root": "run_output_root",
    },
    "upload": {
        "enabled": "upload_google_drive",
        "source": "gdrive_source",
        "remote": "gdrive_remote",
        "dest": "gdrive_dest",
        "folder_id": "gdrive_folder_id",
        "service_account_file": "gdrive_service_account_file",
        "scope": "gdrive_scope",
        "rclone_args": "gdrive_rclone_args",
        "dry_run": "gdrive_dry_run",
    },
    "orchestration": {
        "torch_container": "torch_container",
        "jax_container": "jax_container",
        "use_service_labels": "use_service_labels",
        "label_project": "label_project",
    },
}


@dataclass(frozen=True)
class PipelineStep:
    name: str
    service: str
    command: tuple[str, ...]


def load_config_defaults(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    if not config_path.exists():
        if config_path == DEFAULT_CONFIG:
            return {}
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Pipeline config must be a file: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read pipeline config files. "
            "Install `pyyaml` or run without --config."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Pipeline config must be a YAML mapping: {config_path}")

    defaults = {}
    for section, key_map in CONFIG_KEY_MAP.items():
        raw_section = raw_config.get(section, {})
        if raw_section is None:
            continue
        if not isinstance(raw_section, dict):
            raise ValueError(f"`{section}` in {config_path} must be a mapping")
        for config_key, arg_name in key_map.items():
            if config_key in raw_section and raw_section[config_key] is not None:
                defaults[arg_name] = raw_section[config_key]

    return defaults


def config_path_arg(raw_path: str) -> Path | None:
    if raw_path == "":
        return None
    return Path(raw_path)


def build_parser(config_defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=config_path_arg,
        default=DEFAULT_CONFIG,
        help=(
            "YAML config file used for pipeline defaults. CLI flags override it. "
            "Defaults to `configs/pipeline/default.yaml`; pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--drive",
        default=None,
        help=(
            "Backward-compatible scene id alias. For KITTI-360 this is the drive "
            "id, for example `2013_05_28_drive_0008_sync`."
        ),
    )
    dataset_group = parser.add_argument_group("dataset selection")
    dataset_group.add_argument(
        "--dataset-name",
        choices=("kitti360", "nvidia_ncore"),
        default="kitti360",
        help="Dataset adapter used by the prepare and point-cloud stages.",
    )
    dataset_group.add_argument(
        "--scene-id",
        default=None,
        help="Scene/clip id. Defaults to --drive for backward compatibility.",
    )
    dataset_group.add_argument(
        "--ncore-root",
        type=Path,
        default=None,
        help="Root containing converted NVIDIA PhysicalAI AV NCore clips.",
    )
    dataset_group.add_argument(
        "--camera-id",
        dest="camera_ids",
        action="append",
        default=None,
        help=(
            "NVIDIA NCore camera id to include. Repeat or pass comma-separated "
            "ids. Defaults to camera_front_wide_120fov."
        ),
    )
    dataset_group.add_argument(
        "--point-source",
        choices=("stereo", "lidar", "camera_depth"),
        default=None,
        help="Point source. Defaults to stereo for KITTI-360 and lidar for NVIDIA NCore.",
    )
    dataset_group.add_argument(
        "--camera-depth-pair",
        default=None,
        help="Comma-separated left,right NCore camera ids for camera_depth export.",
    )
    dataset_group.add_argument(
        "--ncore-lidar-id",
        default="lidar_top_360fov",
        help="NCore LiDAR id used by lidar point export and sparse seeding.",
    )
    parser.add_argument(
        "--torch-container",
        default="",
        help=(
            "Optional concrete container name/id for torch steps. When set, "
            "`docker exec` is used instead of `docker compose exec`."
        ),
    )
    parser.add_argument(
        "--jax-container",
        default="",
        help=(
            "Optional concrete container name/id for JAX steps. When set, "
            "`docker exec` is used instead of `docker compose exec`."
        ),
    )
    parser.add_argument(
        "--use-service-labels",
        action="store_true",
        default=True,
        help=(
            "Find sibling containers through Docker Compose labels and run "
            "`docker exec` against them. Enabled by default for the "
            "container-only pipeline entrypoint."
        ),
    )
    parser.add_argument(
        "--label-project",
        default=os.environ.get("VBOGS_COMPOSE_PROJECT", ""),
        help=(
            "Compose project/Portainer stack label used with --use-service-labels. "
            "Defaults to VBOGS_COMPOSE_PROJECT or auto-detects from the current "
            "container when Docker CLI access is available."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=None,
        help=(
            "Optional root for curated run outputs. When set, stage outputs are "
            "derived under `<root>/<drive>/`."
        ),
    )
    parser.add_argument(
        "--skip-local-viewer-export",
        action="store_true",
        help="Do not create the portable local viewer export during the bundle stage.",
    )
    upload_group = parser.add_argument_group("Google Drive upload")
    upload_group.add_argument(
        "--upload-google-drive",
        dest="upload_google_drive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Upload the final curated run zip after all selected stages succeed. "
            "The default source is `<run-output-root>/<drive>.zip`."
        ),
    )
    upload_group.add_argument(
        "--gdrive-source",
        type=Path,
        default=None,
        help="Optional file or directory to upload instead of the curated run zip.",
    )
    upload_group.add_argument(
        "--gdrive-remote",
        default=None,
        help="rclone remote name. Defaults inside upload_google_drive.py.",
    )
    upload_group.add_argument(
        "--gdrive-dest",
        default=None,
        help="Destination path inside the configured Drive remote/root folder.",
    )
    upload_group.add_argument(
        "--gdrive-folder-id",
        default=None,
        help="Google Drive folder id used as rclone's root_folder_id.",
    )
    upload_group.add_argument(
        "--gdrive-service-account-file",
        default=None,
        help="Path to a Google service-account JSON file inside vbogs-pipeline.",
    )
    upload_group.add_argument(
        "--gdrive-scope",
        default=None,
        help="rclone Google Drive scope. Defaults to `drive` in the upload helper.",
    )
    upload_group.add_argument(
        "--gdrive-rclone-args",
        default=None,
        help="Extra arguments appended to the rclone command, parsed with shlex.",
    )
    upload_group.add_argument(
        "--gdrive-dry-run",
        dest="gdrive_dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the rclone upload command without transferring files.",
    )
    parser.add_argument(
        "--start-at",
        choices=STAGES,
        default="prepare",
        help="First stage to run.",
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        default="inspect",
        help="Last stage to run.",
    )
    input_group = parser.add_argument_group("KITTI-360 inputs")
    input_group.add_argument("--raw-root", type=Path, default=None)
    input_group.add_argument("--poses-root", type=Path, default=None)
    input_group.add_argument("--calibration-dir", type=Path, default=None)

    prep_group = parser.add_argument_group("dataset preparation")
    prep_group.add_argument("--frame-step", type=int, default=10)
    prep_group.add_argument("--max-frames", type=int, default=0)
    prep_group.add_argument(
        "--copy-mode",
        choices=("symlink", "copy"),
        default="symlink",
    )
    prep_group.add_argument(
        "--training-cameras",
        choices=("left", "stereo"),
        default="left",
        help=(
            "KITTI-360 RGB cameras used for Octree-AnyGS training. "
            "`left` preserves the legacy image_00-only layout; `stereo` adds "
            "image_01 as a second posed training camera. Ignored by NCore."
        ),
    )
    prep_group.add_argument(
        "--seed-mode",
        choices=("stereo", "lidar", "random"),
        default="stereo",
    )

    train_group = parser.add_argument_group("Octree-AnyGS training")
    train_group.add_argument("--gpu", default="0")
    train_group.add_argument("--resolution", type=int, default=4)
    train_group.add_argument("--iterations", type=int, default=15000)
    train_group.add_argument("--llffhold", type=int, default=8)
    train_group.add_argument(
        "--gaussian-type",
        choices=("implicit3D", "explicit3D"),
        default="implicit3D",
        help=(
            "Octree-AnyGS Gaussian representation. `implicit3D` is the neural "
            "default; `explicit3D` uses explicit SH 3D Gaussians."
        ),
    )
    train_group.add_argument("--feat-dim", type=int, default=16)
    train_group.add_argument("--base-layer", type=int, default=9)
    train_group.add_argument("--visible-threshold", type=float, default=0.02)
    train_group.add_argument(
        "--train-port",
        type=int,
        default=None,
        help=(
            "Octree-AnyGS network GUI port for training. Defaults in the wrapper "
            "to 6009 + GPU index."
        ),
    )
    train_group.add_argument(
        "--write-config-only",
        action="store_true",
        help="Generate the Octree-AnyGS config but skip training.",
    )
    train_group.add_argument(
        "--skip-stack-check",
        action="store_true",
        help=(
            "Skip the Torch CUDA/gsplat preflight before launching Octree-AnyGS "
            "training. Useful only for debugging a stuck preflight."
        ),
    )

    stereo_group = parser.add_argument_group("point-cloud export")
    stereo_group.add_argument(
        "--matcher",
        choices=("sgbm", "raft"),
        default="sgbm",
    )
    stereo_group.add_argument("--pixel-step", type=int, default=1)
    stereo_group.add_argument("--max-points-per-frame", type=int, default=250000)
    stereo_group.add_argument("--write-ply", action="store_true")

    bucket_group = parser.add_argument_group("anchor bucketing")
    bucket_group.add_argument("--model-path", type=Path, default=None)
    bucket_group.add_argument("--bucket-iteration", type=int, default=-1)
    bucket_group.add_argument(
        "--bucket-point-chunk-size",
        type=int,
        default=1_000_000,
        help="Number of world points processed per bucketing chunk.",
    )
    bucket_group.add_argument(
        "--bucket-max-points",
        type=int,
        default=0,
        help="Optional deterministic cap on points used by bucket_points.py; 0 keeps all.",
    )

    fit_group = parser.add_argument_group("VBGS anchor fitting")
    fit_group.add_argument("--jax-device", type=int, default=0)
    fit_group.add_argument(
        "--fit-mode",
        choices=("batched", "loop"),
        default="batched",
    )
    fit_group.add_argument("--batch-size", type=int, default=5000)
    fit_group.add_argument(
        "--batch-buckets",
        default="64,128,256,512,1024,2048,4096,5000",
        help="Comma-separated point-count buckets forwarded to fit_anchors.py.",
    )
    fit_group.add_argument(
        "--no-auto-extend-buckets",
        action="store_true",
        help="Disable automatic dense-tail bucket extension during VBGS fitting.",
    )
    fit_group.add_argument("--vmap-group-size", type=int, default=64)
    fit_group.add_argument(
        "--max-padded-points-per-group",
        type=int,
        default=0,
        help=(
            "Maximum padded anchor-points per batched VBGS fit call. "
            "Defaults to `--vmap-group-size * --batch-size`."
        ),
    )
    fit_group.add_argument("--log-every", type=int, default=100)
    fit_group.add_argument(
        "--max-observed-anchors",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    inspect_group = parser.add_argument_group("anchor fit inspection")
    inspect_group.add_argument(
        "--inspect-top-k",
        type=int,
        default=5,
        help="How many anchors to show per M4b inspection heuristic list.",
    )
    inspect_group.add_argument(
        "--inspect-sample-points",
        type=int,
        default=5,
        help=(
            "How many assigned points to print if --inspect-anchor-id is used."
        ),
    )
    inspect_group.add_argument(
        "--inspect-anchor-id",
        type=int,
        default=None,
        help="Optional explicit anchor id for the final fit inspection stage.",
    )
    inspect_group.add_argument(
        "--inspect-export-ply",
        type=Path,
        default=None,
        help=(
            "Optional PLY export path for --inspect-anchor-id assigned points."
        ),
    )

    uncertainty_group = parser.add_argument_group("uncertainty computation")
    uncertainty_group.add_argument(
        "--uncertainty-u-max",
        type=float,
        default=None,
        help=(
            "Value assigned to unobserved anchors. Defaults to the maximum "
            "finite observed uncertainty."
        ),
    )
    uncertainty_group.add_argument(
        "--uncertainty-no-histogram",
        action="store_true",
        help="Skip writing the M5 uncertainty histogram PNG.",
    )

    map_viz_group = parser.add_argument_group("map-scale uncertainty export")
    map_viz_group.add_argument(
        "--map-viz-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional map-scale PLY output directory. Defaults to "
            "`outputs/uncertainty_maps/<drive>`."
        ),
    )
    map_viz_group.add_argument("--map-viz-vmin", type=float, default=None)
    map_viz_group.add_argument("--map-viz-vmax", type=float, default=None)
    map_viz_group.add_argument("--map-viz-percentile-low", type=float, default=2.0)
    map_viz_group.add_argument("--map-viz-percentile-high", type=float, default=98.0)
    map_viz_group.add_argument(
        "--map-viz-observed-only",
        action="store_true",
        help="Export only observed anchors in the map-scale CloudCompare PLYs.",
    )
    map_viz_group.add_argument(
        "--map-viz-no-split-levels",
        action="store_true",
        help="Only write the combined all-levels map-scale PLY.",
    )
    map_viz_group.add_argument(
        "--map-viz-no-trajectory",
        action="store_true",
        help="Skip the camera trajectory PLY in the map-scale export stage.",
    )

    render_group = parser.add_argument_group("uncertainty rendering")
    render_group.add_argument(
        "--render-split",
        choices=("train", "test", "both"),
        default="both",
        help="Camera split rendered by the final diagnostic stage.",
    )
    render_group.add_argument(
        "--render-resolution",
        type=int,
        default=2,
        help=(
            "Octree-AnyGS image divisor/target width for diagnostic renders. "
            "Smaller divisors produce higher-resolution views."
        ),
    )
    render_group.add_argument(
        "--render-max-views",
        type=int,
        default=0,
        help="Optional cap per split for render smoke tests. `0` renders all views.",
    )
    render_group.add_argument(
        "--render-colormap",
        default="turbo",
        help="Matplotlib colormap for rendered uncertainty heatmaps.",
    )
    render_group.add_argument("--render-vmin", type=float, default=None)
    render_group.add_argument("--render-vmax", type=float, default=None)
    render_group.add_argument(
        "--render-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional render output root. Defaults to "
            "`outputs/uncertainty_views/<drive>` in the Torch container."
        ),
    )

    nbv_group = parser.add_argument_group("next-best-view scoring")
    nbv_group.add_argument(
        "--nbv-candidate-source",
        choices=("test", "train", "lattice"),
        default="test",
        help="Candidate camera set passed to score_nbv.py.",
    )
    nbv_group.add_argument(
        "--nbv-max-candidates",
        type=int,
        default=0,
        help="Optional candidate cap for NBV scoring. `0` scores all candidates.",
    )
    nbv_group.add_argument(
        "--nbv-top-k",
        type=int,
        default=10,
        help="Number of ranked NBV candidates saved in nbv_scores.json.",
    )
    nbv_group.add_argument(
        "--nbv-save-top-images",
        type=int,
        default=5,
        help="Number of top uncertainty/alpha arrays saved for NBV visualization.",
    )
    nbv_group.add_argument(
        "--nbv-force-all-levels",
        action="store_true",
        help="Force all Octree-AnyGS levels active while scoring NBV candidates.",
    )
    nbv_group.add_argument(
        "--nbv-output-dir",
        type=Path,
        default=None,
        help="Optional NBV output directory. Defaults to `data/m6/<drive>`.",
    )
    parser.set_defaults(**(config_defaults or {}))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=config_path_arg, default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = pre_args.config
    config_defaults = load_config_defaults(config_path)

    parser = build_parser(config_defaults)
    args = parser.parse_args(argv)
    if not (args.scene_id or args.drive):
        parser.error(
            "--scene-id is required unless --drive or `pipeline.drive` is set in the config"
        )
    return args


def maybe_path_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for arg_name, flag in (
        ("raw_root", "--raw-root"),
        ("poses_root", "--poses-root"),
        ("calibration_dir", "--calibration-dir"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            result.extend([flag, str(value)])
    return result


def maybe_option(flag: str, value: object | None) -> list[str]:
    if value is None or value == "":
        return []
    return [flag, str(value)]


def scene_id_arg(args: argparse.Namespace) -> str:
    scene = args.scene_id or args.drive
    if not scene:
        raise ValueError("A scene id is required")
    return str(scene)


def effective_point_source(args: argparse.Namespace) -> str:
    if args.point_source:
        return str(args.point_source)
    return "stereo" if args.dataset_name == "kitti360" else "lidar"


def camera_id_args(camera_ids: object | None) -> list[str]:
    if camera_ids is None or camera_ids == "":
        return []
    if isinstance(camera_ids, (str, Path)):
        values = [str(camera_ids)]
    else:
        values = [str(value) for value in camera_ids]
    result: list[str] = []
    for value in values:
        result.extend(["--camera-id", value])
    return result


def run_output_dir(args: argparse.Namespace) -> Path | None:
    if args.run_output_root is None:
        return None
    return Path(args.run_output_root) / scene_id_arg(args)


def derived_output_dir(
    explicit: Path | None,
    base_dir: Path | None,
    *parts: str,
) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    if base_dir is None:
        return None
    return base_dir.joinpath(*parts)


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    scene = scene_id_arg(args)
    dataset_path = f"/data/COLMAP/{scene}"
    selection_metadata = f"{dataset_path}/metadata.json"
    bucket_root = f"data/m4/{scene}"
    run_dir = run_output_dir(args)
    map_viz_output_dir = derived_output_dir(
        args.map_viz_output_dir,
        run_dir,
        "pointclouds",
        "anchors",
    )
    render_output_dir = derived_output_dir(args.render_output_dir, run_dir, "views")
    nbv_output_dir = derived_output_dir(args.nbv_output_dir, run_dir, "nbv")

    if args.dataset_name == "kitti360":
        prepare_cmd = (
            "python",
            "scripts/prepare_kitti360_colmap.py",
            "--drive",
            scene,
            "--frame-step",
            str(args.frame_step),
            "--max-frames",
            str(args.max_frames),
            "--copy-mode",
            args.copy_mode,
            "--training-cameras",
            args.training_cameras,
            "--seed-mode",
            "stereo" if args.seed_mode == "lidar" else args.seed_mode,
            *maybe_path_args(args),
        )
    else:
        prepare_cmd = (
            "python",
            "scripts/prepare_nvidia_ncore_colmap.py",
            "--scene-id",
            scene,
            "--frame-step",
            str(args.frame_step),
            "--max-frames",
            str(args.max_frames),
            "--copy-mode",
            args.copy_mode,
            "--seed-mode",
            "lidar" if args.seed_mode == "stereo" else args.seed_mode,
            "--lidar-id",
            args.ncore_lidar_id,
            *maybe_option("--ncore-root", args.ncore_root),
            *camera_id_args(args.camera_ids),
        )

    train_cmd = (
        "python",
        "scripts/train_octree_anygs.py",
        "--dataset-path",
        dataset_path,
        "--scene-name",
        scene,
        "--dataset-name",
        args.dataset_name,
        "--gpu",
        str(args.gpu),
        "--resolution",
        str(args.resolution),
        "--iterations",
        str(args.iterations),
        "--llffhold",
        str(args.llffhold),
        "--gaussian-type",
        args.gaussian_type,
        "--feat-dim",
        str(args.feat_dim),
        "--base-layer",
        str(args.base_layer),
        "--visible-threshold",
        str(args.visible_threshold),
        *maybe_option("--port", args.train_port),
        *(("--write-config-only",) if args.write_config_only else ()),
        *(("--skip-stack-check",) if args.skip_stack_check else ()),
    )

    point_cmd = (
        "python",
        "scripts/export_points_world.py",
        "--dataset-name",
        args.dataset_name,
        "--scene-id",
        scene,
        "--point-source",
        effective_point_source(args),
        "--selection-metadata",
        selection_metadata,
        "--matcher",
        args.matcher,
        "--pixel-step",
        str(args.pixel_step),
        "--max-points-per-frame",
        str(args.max_points_per_frame),
        "--max-frames",
        str(args.max_frames),
        *(("--write-ply",) if args.write_ply else ()),
        *maybe_path_args(args),
        *maybe_option("--ncore-root", args.ncore_root),
        *camera_id_args(args.camera_ids),
        *maybe_option("--camera-depth-pair", args.camera_depth_pair),
        *maybe_option("--lidar-id", args.ncore_lidar_id),
    )

    bucket_cmd = (
        "python",
        "scripts/bucket_points.py",
        "--drive",
        scene,
        "--iteration",
        str(args.bucket_iteration),
        "--point-chunk-size",
        str(args.bucket_point_chunk_size),
        "--max-points",
        str(args.bucket_max_points),
        *maybe_option("--model-path", args.model_path),
    )

    fit_cmd = (
        "python",
        "scripts/fit_anchors.py",
        "--drive",
        scene,
        "--device",
        str(args.jax_device),
        "--fit-mode",
        args.fit_mode,
        "--batch-size",
        str(args.batch_size),
        "--batch-buckets",
        args.batch_buckets,
        "--vmap-group-size",
        str(args.vmap_group_size),
        *(("--no-auto-extend-buckets",) if args.no_auto_extend_buckets else ()),
        "--max-padded-points-per-group",
        str(args.max_padded_points_per_group),
        "--log-every",
        str(args.log_every),
    )

    posterior_name = "anchor_posterior.npz"
    inspect_cmd = (
        "python",
        "scripts/inspect_anchor_fits.py",
        "--drive",
        scene,
        "--bucket-root",
        bucket_root,
        "--posterior",
        f"{bucket_root}/{posterior_name}",
        "--top-k",
        str(args.inspect_top_k),
        "--sample-points",
        str(args.inspect_sample_points),
        *maybe_option("--anchor-id", args.inspect_anchor_id),
        *maybe_option("--export-ply", args.inspect_export_ply),
    )

    uncertainty_cmd = (
        "python",
        "scripts/compute_uncertainty.py",
        "--drive",
        scene,
        "--bucket-root",
        bucket_root,
        "--posterior",
        f"{bucket_root}/{posterior_name}",
        *maybe_option("--u-max", args.uncertainty_u_max),
        *(("--no-histogram",) if args.uncertainty_no_histogram else ()),
    )

    map_viz_cmd = (
        "python",
        "scripts/export_uncertainty_map.py",
        "--drive",
        scene,
        "--bucket-root",
        bucket_root,
        "--posterior",
        f"{bucket_root}/{posterior_name}",
        "--selection-metadata",
        selection_metadata,
        "--percentile-low",
        str(args.map_viz_percentile_low),
        "--percentile-high",
        str(args.map_viz_percentile_high),
        *maybe_option("--poses-root", args.poses_root),
        *maybe_option("--output-dir", map_viz_output_dir),
        *maybe_option("--vmin", args.map_viz_vmin),
        *maybe_option("--vmax", args.map_viz_vmax),
        *(("--observed-only",) if args.map_viz_observed_only else ()),
        *(("--no-split-levels",) if args.map_viz_no_split_levels else ()),
        *(("--no-trajectory",) if args.map_viz_no_trajectory else ()),
    )

    render_cmd = (
        "python",
        "scripts/render_uncertainty_views.py",
        "--drive",
        scene,
        *maybe_option("--model-path", args.model_path),
        "--uncertainty",
        f"{bucket_root}/U.npy",
        "--iteration",
        str(args.bucket_iteration),
        "--split",
        args.render_split,
        *maybe_option("--resolution", args.render_resolution),
        "--max-views",
        str(args.render_max_views),
        "--colormap",
        args.render_colormap,
        *maybe_option("--vmin", args.render_vmin),
        *maybe_option("--vmax", args.render_vmax),
        *maybe_option("--output-dir", render_output_dir),
    )

    nbv_cmd = (
        "python",
        "scripts/score_nbv.py",
        "--drive",
        scene,
        *maybe_option("--model-path", args.model_path),
        "--u-path",
        f"{bucket_root}/U.npy",
        "--iteration",
        str(args.bucket_iteration),
        "--candidate-source",
        args.nbv_candidate_source,
        "--max-candidates",
        str(args.nbv_max_candidates),
        "--top-k",
        str(args.nbv_top_k),
        "--save-top-images",
        str(args.nbv_save_top_images),
        *maybe_option("--output-root", nbv_output_dir),
        *(("--force-all-levels",) if args.nbv_force_all_levels else ()),
    )

    nbv_viz_cmd = (
        "python",
        "scripts/visualize_m6.py",
        "--drive",
        scene,
        *maybe_option("--m6-root", nbv_output_dir),
        *maybe_option("--output-dir", nbv_output_dir / "viz" if nbv_output_dir else None),
    )

    bundle_cmd = (
        "python",
        "scripts/bundle_run_outputs.py",
        "--drive",
        scene,
        *maybe_option("--run-output-dir", run_dir),
        *maybe_option("--model-path", args.model_path),
        *maybe_option("--map-viz-output-dir", map_viz_output_dir),
        *maybe_option("--render-output-dir", render_output_dir),
        *maybe_option("--nbv-output-dir", nbv_output_dir),
        "--viewer-export-iteration",
        str(args.bucket_iteration),
        *(("--skip-local-viewer-export",) if args.skip_local_viewer_export else ()),
    )

    return [
        PipelineStep("prepare", TORCH_SERVICE, prepare_cmd),
        PipelineStep("train", TORCH_SERVICE, train_cmd),
        PipelineStep("stereo", TORCH_SERVICE, point_cmd),
        PipelineStep("bucket", TORCH_SERVICE, bucket_cmd),
        PipelineStep("fit", JAX_SERVICE, fit_cmd),
        PipelineStep("inspect", JAX_SERVICE, inspect_cmd),
        PipelineStep("uncertainty", JAX_SERVICE, uncertainty_cmd),
        PipelineStep("map-viz", TORCH_SERVICE, map_viz_cmd),
        PipelineStep("render", TORCH_SERVICE, render_cmd),
        PipelineStep("nbv", TORCH_SERVICE, nbv_cmd),
        PipelineStep("nbv-viz", TORCH_SERVICE, nbv_viz_cmd),
        PipelineStep("bundle", TORCH_SERVICE, bundle_cmd),
    ]


def selected_steps(
    steps: Sequence[PipelineStep],
    start_at: str,
    stop_after: str,
) -> list[PipelineStep]:
    start_idx = STAGES.index(start_at)
    stop_idx = STAGES.index(stop_after)
    if start_idx > stop_idx:
        raise ValueError("--start-at must be earlier than or equal to --stop-after")
    selected_names = set(STAGES[start_idx : stop_idx + 1])
    return [step for step in steps if step.name in selected_names]


def container_override(args: argparse.Namespace, service: str) -> str:
    if service == TORCH_SERVICE:
        return args.torch_container
    if service == JAX_SERVICE:
        return args.jax_container
    return ""


def current_container_project() -> str:
    container_id = os.environ.get("HOSTNAME", "")
    if not container_id:
        return ""
    try:
        completed = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{ index .Config.Labels \"com.docker.compose.project\" }}",
                container_id,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    project = completed.stdout.strip()
    if project == "<no value>":
        return ""
    return project


def running_inside_container() -> bool:
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False
    return any(
        marker in cgroup
        for marker in ("docker", "containerd", "kubepods", "libpod")
    )


def require_container_runtime() -> None:
    if running_inside_container():
        return
    raise RuntimeError(
        "Host-side pipeline orchestration is disabled. Enter vbogs-pipeline "
        "with `./dc_bash.sh`, then run `scripts/run_pipeline.sh ...`."
    )


def resolve_service_container(
    service: str,
    *,
    project: str,
    dry_run: bool,
) -> str:
    if dry_run:
        if project:
            return f"<{project}:{service}-container-by-label>"
        return f"<{service}-container-by-label>"

    filters = [
        "--filter",
        f"label=com.docker.compose.service={service}",
        "--filter",
        "status=running",
    ]
    if project:
        filters.extend(["--filter", f"label=com.docker.compose.project={project}"])

    cmd = ["docker", "ps", "-q", *filters]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker CLI is required for --use-service-labels") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to resolve container for service {service}: {exc.stderr.strip()}"
        ) from exc

    matches = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(matches) != 1:
        project_hint = f" in project {project}" if project else ""
        raise RuntimeError(
            f"Expected exactly one running container for service {service}{project_hint}; "
            f"found {len(matches)}. Pass --label-project or explicit container names."
        )
    return matches[0]


def exec_prefix(args: argparse.Namespace, service: str) -> list[str]:
    container = container_override(args, service)
    if container:
        return ["docker", "exec", "-i", "-w", "/workspace/VBOGS", container]
    if args.use_service_labels:
        project = args.label_project or current_container_project()
        container = resolve_service_container(service, project=project, dry_run=args.dry_run)
        return ["docker", "exec", "-i", "-w", "/workspace/VBOGS", container]
    raise RuntimeError(
        "No container resolution mode is enabled. Use scripts/run_pipeline.sh "
        "inside vbogs-pipeline, or pass explicit --torch-container and "
        "--jax-container values."
    )


def shell_exec_command(script: str) -> tuple[str, ...]:
    return ("sh", "-lc", script)


def run_command(cmd: Sequence[str], *, dry_run: bool) -> None:
    printable = shlex.join(cmd)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_upload_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/upload_google_drive.py",
        "--config",
        str(args.config) if args.config is not None else "",
        "--drive",
        scene_id_arg(args),
    ]

    if args.run_output_root is not None:
        cmd.extend(["--run-output-root", str(args.run_output_root)])

    for arg_name, flag in (
        ("gdrive_source", "--source"),
        ("gdrive_remote", "--remote"),
        ("gdrive_dest", "--dest"),
        ("gdrive_folder_id", "--folder-id"),
        ("gdrive_service_account_file", "--service-account-file"),
        ("gdrive_scope", "--scope"),
        ("gdrive_rclone_args", "--rclone-args"),
    ):
        value = getattr(args, arg_name)
        if value is not None and value != "":
            cmd.extend([flag, str(value)])

    if args.dry_run or args.gdrive_dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    args = parse_args()
    require_container_runtime()
    steps = selected_steps(build_steps(args), args.start_at, args.stop_after)

    print(f"Dataset: {args.dataset_name}")
    print(f"Scene: {scene_id_arg(args)}")
    print("Stages: " + ", ".join(step.name for step in steps))
    if args.upload_google_drive:
        print("Upload: Google Drive after successful stages")

    for step in steps:
        print(f"\n=== {step.name} ({step.service}) ===", flush=True)
        run_command([*exec_prefix(args, step.service), *step.command], dry_run=args.dry_run)

    if args.upload_google_drive:
        print("\n=== upload (vbogs-pipeline) ===", flush=True)
        run_command(build_upload_command(args), dry_run=args.dry_run)

    print("\nPipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
