#!/usr/bin/env python3

"""Run the implemented VBOGS pipeline for one KITTI-360 drive.

This is an orchestration script for the two-container Docker / Portainer stack.
It can run from the host or from a stack-contained `vbogs-pipeline` service.
The framework boundary stays explicit:

- `vbogs-torch` runs dataset prep, Octree-AnyGS training, stereo export, and
  point-to-anchor bucketing.
- `vbogs-jax` runs per-anchor VBGS fitting and fit inspection.

Data moves only through the shared stack volumes mounted at the same paths in
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
)
TORCH_SERVICE = "vbogs-torch"
JAX_SERVICE = "vbogs-jax"
DEFAULT_CONFIG = Path("pipeline_config.yaml")
CONFIG_KEY_MAP = {
    "pipeline": {
        "drive": "drive",
        "start_at": "start_at",
        "stop_after": "stop_after",
        "dry_run": "dry_run",
        "skip_up": "skip_up",
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
        "seed_mode": "seed_mode",
    },
    "train": {
        "gpu": "gpu",
        "resolution": "resolution",
        "iterations": "iterations",
        "llffhold": "llffhold",
        "feat_dim": "feat_dim",
        "base_layer": "base_layer",
        "visible_threshold": "visible_threshold",
        "write_config_only": "write_config_only",
    },
    "stereo": {
        "matcher": "matcher",
        "pixel_step": "pixel_step",
        "max_points_per_frame": "max_points_per_frame",
        "write_ply": "write_ply",
    },
    "bucket": {
        "model_path": "model_path",
        "bucket_iteration": "bucket_iteration",
    },
    "fit": {
        "jax_device": "jax_device",
        "fit_mode": "fit_mode",
        "batch_size": "batch_size",
        "vmap_group_size": "vmap_group_size",
        "log_every": "log_every",
        "max_observed_anchors": "max_observed_anchors",
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
    },
    "render": {
        "split": "render_split",
        "max_views": "render_max_views",
        "colormap": "render_colormap",
        "vmin": "render_vmin",
        "vmax": "render_vmax",
        "output_dir": "render_output_dir",
    },
    "orchestration": {
        "compose_command": "compose_command",
        "compose_file": "compose_file",
        "project_name": "project_name",
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
            "Defaults to `pipeline_config.yaml`; pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--drive",
        default=None,
        help="KITTI-360 drive id, for example `2013_05_28_drive_0008_sync`.",
    )
    parser.add_argument(
        "--compose-command",
        default="docker compose",
        help="Compose command used on the host. Defaults to `docker compose`.",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path("docker-compose.yml"),
        help="Compose file for the VBOGS stack.",
    )
    parser.add_argument(
        "--project-name",
        default="",
        help="Optional compose/Portainer stack project name passed as `-p`.",
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
        help=(
            "Find sibling containers through Docker Compose labels and run "
            "`docker exec` against them. This is the mode used by the "
            "stack-contained `vbogs-pipeline` service."
        ),
    )
    parser.add_argument(
        "--label-project",
        default=os.environ.get("VBOGS_COMPOSE_PROJECT", ""),
        help=(
            "Compose project/Portainer stack label used with --use-service-labels. "
            "Defaults to VBOGS_COMPOSE_PROJECT or auto-detects from this container."
        ),
    )
    parser.add_argument(
        "--skip-up",
        action="store_true",
        help="Do not run `docker compose up -d` before executing stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
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
        "--seed-mode",
        choices=("stereo", "random"),
        default="stereo",
    )

    train_group = parser.add_argument_group("Octree-AnyGS training")
    train_group.add_argument("--gpu", default="0")
    train_group.add_argument("--resolution", type=int, default=4)
    train_group.add_argument("--iterations", type=int, default=15000)
    train_group.add_argument("--llffhold", type=int, default=8)
    train_group.add_argument("--feat-dim", type=int, default=16)
    train_group.add_argument("--base-layer", type=int, default=9)
    train_group.add_argument("--visible-threshold", type=float, default=0.02)
    train_group.add_argument(
        "--write-config-only",
        action="store_true",
        help="Generate the Octree-AnyGS config but skip training.",
    )

    stereo_group = parser.add_argument_group("stereo point cloud")
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

    fit_group = parser.add_argument_group("VBGS anchor fitting")
    fit_group.add_argument("--jax-device", type=int, default=0)
    fit_group.add_argument(
        "--fit-mode",
        choices=("batched", "loop"),
        default="batched",
    )
    fit_group.add_argument("--batch-size", type=int, default=5000)
    fit_group.add_argument("--vmap-group-size", type=int, default=64)
    fit_group.add_argument("--log-every", type=int, default=100)
    fit_group.add_argument(
        "--max-observed-anchors",
        type=int,
        default=0,
        help=(
            "Optional smoke-test cap for M4b. Leave at 0 for the full fit."
        ),
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

    render_group = parser.add_argument_group("uncertainty rendering")
    render_group.add_argument(
        "--render-split",
        choices=("train", "test", "both"),
        default="both",
        help="Camera split rendered by the final diagnostic stage.",
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
    if not args.drive:
        parser.error("--drive is required unless `pipeline.drive` is set in the config")
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


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    dataset_path = f"/data/COLMAP/{args.drive}"
    selection_metadata = f"{dataset_path}/metadata.json"
    bucket_root = f"data/m4/{args.drive}"

    prepare_cmd = (
        "python",
        "scripts/prepare_kitti360_colmap.py",
        "--drive",
        args.drive,
        "--frame-step",
        str(args.frame_step),
        "--max-frames",
        str(args.max_frames),
        "--copy-mode",
        args.copy_mode,
        "--seed-mode",
        args.seed_mode,
        *maybe_path_args(args),
    )

    train_cmd = (
        "python",
        "scripts/train_octree_anygs.py",
        "--dataset-path",
        dataset_path,
        "--scene-name",
        args.drive,
        "--gpu",
        str(args.gpu),
        "--resolution",
        str(args.resolution),
        "--iterations",
        str(args.iterations),
        "--llffhold",
        str(args.llffhold),
        "--feat-dim",
        str(args.feat_dim),
        "--base-layer",
        str(args.base_layer),
        "--visible-threshold",
        str(args.visible_threshold),
        *(("--write-config-only",) if args.write_config_only else ()),
    )

    stereo_cmd = (
        "python",
        "scripts/stereo_to_pointcloud.py",
        "--drive",
        args.drive,
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
    )

    bucket_cmd = (
        "python",
        "scripts/bucket_points.py",
        "--drive",
        args.drive,
        "--iteration",
        str(args.bucket_iteration),
        *maybe_option("--model-path", args.model_path),
    )

    fit_cmd = (
        "python",
        "scripts/fit_anchors.py",
        "--drive",
        args.drive,
        "--device",
        str(args.jax_device),
        "--fit-mode",
        args.fit_mode,
        "--batch-size",
        str(args.batch_size),
        "--vmap-group-size",
        str(args.vmap_group_size),
        "--log-every",
        str(args.log_every),
        "--max-observed-anchors",
        str(args.max_observed_anchors),
    )

    posterior_name = (
        "anchor_posterior.smoke.npz"
        if args.max_observed_anchors > 0
        else "anchor_posterior.npz"
    )
    inspect_cmd = (
        "python",
        "scripts/inspect_anchor_fits.py",
        "--drive",
        args.drive,
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
        args.drive,
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
        args.drive,
        "--bucket-root",
        bucket_root,
        "--posterior",
        f"{bucket_root}/{posterior_name}",
        "--percentile-low",
        str(args.map_viz_percentile_low),
        "--percentile-high",
        str(args.map_viz_percentile_high),
        *maybe_option("--output-dir", args.map_viz_output_dir),
        *maybe_option("--vmin", args.map_viz_vmin),
        *maybe_option("--vmax", args.map_viz_vmax),
        *(("--observed-only",) if args.map_viz_observed_only else ()),
        *(("--no-split-levels",) if args.map_viz_no_split_levels else ()),
    )

    render_cmd = (
        "python",
        "scripts/render_uncertainty_views.py",
        "--drive",
        args.drive,
        "--split",
        args.render_split,
        "--max-views",
        str(args.render_max_views),
        "--colormap",
        args.render_colormap,
        *maybe_option("--vmin", args.render_vmin),
        *maybe_option("--vmax", args.render_vmax),
        *maybe_option("--output-dir", args.render_output_dir),
    )

    return [
        PipelineStep("prepare", TORCH_SERVICE, prepare_cmd),
        PipelineStep("train", TORCH_SERVICE, train_cmd),
        PipelineStep("stereo", TORCH_SERVICE, stereo_cmd),
        PipelineStep("bucket", TORCH_SERVICE, bucket_cmd),
        PipelineStep("fit", JAX_SERVICE, fit_cmd),
        PipelineStep("inspect", JAX_SERVICE, inspect_cmd),
        PipelineStep("uncertainty", JAX_SERVICE, uncertainty_cmd),
        PipelineStep("map-viz", TORCH_SERVICE, map_viz_cmd),
        PipelineStep("render", TORCH_SERVICE, render_cmd),
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


def compose_base(args: argparse.Namespace) -> list[str]:
    base = shlex.split(args.compose_command)
    if args.compose_file:
        base.extend(["-f", str(args.compose_file)])
    if args.project_name:
        base.extend(["-p", args.project_name])
    return base


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
    return [*compose_base(args), "exec", "-T", service]


def shell_exec_command(script: str) -> tuple[str, ...]:
    return ("sh", "-lc", script)


def run_command(cmd: Sequence[str], *, dry_run: bool) -> None:
    printable = shlex.join(cmd)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_optional_up(args: argparse.Namespace, steps: Sequence[PipelineStep]) -> None:
    if args.skip_up or args.use_service_labels:
        return
    services = sorted(
        {step.service for step in steps if not container_override(args, step.service)}
    )
    if not services:
        return
    run_command([*compose_base(args), "up", "-d", *services], dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    steps = selected_steps(build_steps(args), args.start_at, args.stop_after)

    print(f"Drive: {args.drive}")
    print("Stages: " + ", ".join(step.name for step in steps))
    run_optional_up(args, steps)

    for step in steps:
        print(f"\n=== {step.name} ({step.service}) ===", flush=True)
        run_command([*exec_prefix(args, step.service), *step.command], dry_run=args.dry_run)

    print("\nPipeline completed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
