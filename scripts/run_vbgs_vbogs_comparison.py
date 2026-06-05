#!/usr/bin/env python3

"""Run the VBGS-vs-VBOGS uncertainty-quality comparison workflow."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vbgs_kitti_baseline import (
    current_container_project,
    resolve_service_container,
)


TORCH_SERVICE = "vbogs-torch"
JAX_SERVICE = "vbogs-jax"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "vbgs_comparison"


@dataclass(frozen=True)
class ComparisonStep:
    name: str
    service: str
    command: tuple[str, ...]


def parse_k_sweep(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--k-sweep must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("--k-sweep values must be positive")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    parser.add_argument("--use-service-labels", action="store_true")
    parser.add_argument(
        "--label-project",
        default=os.environ.get("VBOGS_COMPOSE_PROJECT", ""),
    )
    parser.add_argument("--torch-container", default="")
    parser.add_argument("--jax-container", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--points-world", type=Path, default=None)
    parser.add_argument("--selection-metadata", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--bucket-iteration", type=int, default=-1)
    parser.add_argument("--bucket-point-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--jax-device", type=int, default=0)
    parser.add_argument(
        "--fit-mode",
        choices=("batched", "loop"),
        default="batched",
    )
    parser.add_argument("--fit-batch-size", type=int, default=5000)
    parser.add_argument("--fit-vmap-group-size", type=int, default=64)
    parser.add_argument("--fit-log-every", type=int, default=500)
    parser.add_argument("--vbgs-batch-size", type=int, default=500)
    parser.add_argument("--vbgs-seed", type=int, default=0)
    parser.add_argument("--k-sweep", default="1000,3000,10000,30000")
    parser.add_argument("--no-reassign", action="store_true")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--render-split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--render-resolution", type=int, default=2)
    parser.add_argument("--render-max-views", type=int, default=5)
    parser.add_argument("--no-map-viz", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def resolve_output_root(drive: str, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root
    return DEFAULT_OUTPUT_ROOT / drive


def maybe_option(flag: str, value: object | None) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    return (flag, str(value))


def build_steps(args: argparse.Namespace) -> list[ComparisonStep]:
    root = resolve_output_root(args.drive, args.output_root)
    train_bucket = root / "m4_train"
    eval_bucket = root / "m4_eval"
    global_root = root / "vbgs_global"
    train_points = root / "points_train.npz"
    eval_points = root / "points_eval.npz"
    posterior = train_bucket / "anchor_posterior.npz"
    vbogs_u = train_bucket / "U.npy"
    selection_metadata = (
        args.selection_metadata or Path("/data/COLMAP") / args.drive / "metadata.json"
    )

    steps: list[ComparisonStep] = [
        ComparisonStep(
            "split-points",
            TORCH_SERVICE,
            (
                "python",
                "scripts/split_vbgs_vbogs_points.py",
                "--drive",
                args.drive,
                "--selection-metadata",
                str(selection_metadata),
                "--output-root",
                str(root),
                "--llffhold",
                str(args.llffhold),
                *maybe_option("--points-world", args.points_world),
            ),
        ),
        ComparisonStep(
            "bucket-train",
            TORCH_SERVICE,
            (
                "python",
                "scripts/bucket_points.py",
                "--drive",
                args.drive,
                "--points-world",
                str(train_points),
                "--output-root",
                str(train_bucket),
                "--iteration",
                str(args.bucket_iteration),
                "--point-chunk-size",
                str(args.bucket_point_chunk_size),
                "--max-points",
                "0",
                *maybe_option("--model-path", args.model_path),
            ),
        ),
        ComparisonStep(
            "bucket-eval",
            TORCH_SERVICE,
            (
                "python",
                "scripts/bucket_points.py",
                "--drive",
                args.drive,
                "--points-world",
                str(eval_points),
                "--output-root",
                str(eval_bucket),
                "--iteration",
                str(args.bucket_iteration),
                "--point-chunk-size",
                str(args.bucket_point_chunk_size),
                "--max-points",
                "0",
                *maybe_option("--model-path", args.model_path),
            ),
        ),
        ComparisonStep(
            "fit-vbogs",
            JAX_SERVICE,
            (
                "python",
                "scripts/fit_anchors.py",
                "--drive",
                args.drive,
                "--bucket-root",
                str(train_bucket),
                "--output-root",
                str(train_bucket),
                "--device",
                str(args.jax_device),
                "--fit-mode",
                args.fit_mode,
                "--batch-size",
                str(args.fit_batch_size),
                "--vmap-group-size",
                str(args.fit_vmap_group_size),
                "--log-every",
                str(args.fit_log_every),
            ),
        ),
        ComparisonStep(
            "compute-vbogs-uncertainty",
            JAX_SERVICE,
            (
                "python",
                "scripts/compute_uncertainty.py",
                "--drive",
                args.drive,
                "--bucket-root",
                str(train_bucket),
                "--posterior",
                str(posterior),
            ),
        ),
    ]

    for k in parse_k_sweep(args.k_sweep):
        k_root = global_root / f"K_{k}"
        cmd = [
            "python",
            "scripts/train_vbgs_kitti_baseline.py",
            "--drive",
            args.drive,
            "--input-mode",
            "bucket",
            "--points-norm",
            str(train_bucket / "points_norm.npz"),
            "--norm-params",
            str(train_bucket / "norm_params.json"),
            "--output-root",
            str(k_root),
            "--n-components",
            str(k),
            "--batch-size",
            str(args.vbgs_batch_size),
            "--seed",
            str(args.vbgs_seed),
            "--device",
            str(args.jax_device),
            "--project-anchors",
            "--pts-by-anchor",
            str(train_bucket / "pts_by_anchor.npz"),
            "--eval-bucket-root",
            str(eval_bucket),
        ]
        if args.no_reassign:
            cmd.append("--no-reassign")
        steps.append(ComparisonStep(f"global-vbgs-K-{k}", JAX_SERVICE, tuple(cmd)))

    steps.append(
        ComparisonStep(
            "comparison-report",
            JAX_SERVICE,
            (
                "python",
                "scripts/compare_vbgs_vbogs_uncertainty.py",
                "--drive",
                args.drive,
                "--comparison-root",
                str(root),
                "--train-bucket-root",
                str(train_bucket),
                "--eval-bucket-root",
                str(eval_bucket),
                "--vbogs-u",
                str(vbogs_u),
                "--baseline-root",
                str(global_root),
                "--top-n",
                str(args.top_n),
                *(("--no-plots",) if args.no_plots else ()),
            ),
        )
    )

    if not args.no_map_viz:
        steps.append(
            ComparisonStep(
                "map-vbogs",
                TORCH_SERVICE,
                (
                    "python",
                    "scripts/export_uncertainty_map.py",
                    "--drive",
                    args.drive,
                    "--bucket-root",
                    str(train_bucket),
                    "--uncertainty",
                    str(vbogs_u),
                    "--posterior",
                    str(posterior),
                    "--output-dir",
                    str(root / "maps" / "vbogs"),
                ),
            )
        )
        for k in parse_k_sweep(args.k_sweep):
            steps.append(
                ComparisonStep(
                    f"map-global-vbgs-K-{k}",
                    TORCH_SERVICE,
                    (
                        "python",
                        "scripts/export_uncertainty_map.py",
                        "--drive",
                        args.drive,
                        "--bucket-root",
                        str(train_bucket),
                        "--uncertainty",
                        str(global_root / f"K_{k}" / "U_baseline.npy"),
                        "--posterior",
                        str(posterior),
                        "--output-dir",
                        str(root / "maps" / f"global_vbgs_K_{k}"),
                    ),
                )
            )

    if not args.no_render:
        render_common = (
            "--split",
            args.render_split,
            "--resolution",
            str(args.render_resolution),
            "--max-views",
            str(args.render_max_views),
            *maybe_option("--model-path", args.model_path),
        )
        steps.append(
            ComparisonStep(
                "render-vbogs",
                TORCH_SERVICE,
                (
                    "python",
                    "scripts/render_uncertainty_views.py",
                    "--drive",
                    args.drive,
                    "--uncertainty",
                    str(vbogs_u),
                    "--output-dir",
                    str(root / "views" / "vbogs"),
                    *render_common,
                ),
            )
        )
        for k in parse_k_sweep(args.k_sweep):
            steps.append(
                ComparisonStep(
                    f"render-global-vbgs-K-{k}",
                    TORCH_SERVICE,
                    (
                        "python",
                        "scripts/render_uncertainty_views.py",
                        "--drive",
                        args.drive,
                        "--uncertainty",
                        str(global_root / f"K_{k}" / "U_baseline.npy"),
                        "--output-dir",
                        str(root / "views" / f"global_vbgs_K_{k}"),
                        *render_common,
                    ),
                )
            )
    return steps


def container_override(args: argparse.Namespace, service: str) -> str:
    if service == TORCH_SERVICE:
        return args.torch_container
    if service == JAX_SERVICE:
        return args.jax_container
    return ""


def exec_prefix(args: argparse.Namespace, service: str) -> list[str]:
    container = container_override(args, service)
    if not container:
        if not args.use_service_labels:
            raise RuntimeError(
                "Pass --use-service-labels from vbogs-pipeline, or pass "
                "explicit --torch-container and --jax-container values."
            )
        project = args.label_project or current_container_project()
        container = resolve_service_container(
            service,
            project=project,
            dry_run=args.dry_run,
        )
    return ["docker", "exec", "-i", "-w", "/workspace/VBOGS", container]


def run_printed(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    steps = build_steps(args)
    print(f"Drive: {args.drive}")
    print(f"Output: {resolve_output_root(args.drive, args.output_root)}")
    print("Steps: " + ", ".join(step.name for step in steps))
    for step in steps:
        print(f"\n=== {step.name} ({step.service}) ===", flush=True)
        run_printed(
            [*exec_prefix(args, step.service), *step.command],
            dry_run=args.dry_run,
        )
    print("\nComparison completed.")


if __name__ == "__main__":
    main()
