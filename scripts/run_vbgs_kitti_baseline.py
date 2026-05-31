#!/usr/bin/env python3

"""Launch the original VBGS KITTI baseline in the sibling JAX container."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence


JAX_SERVICE = "vbogs-jax"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0007_sync",
        help="KITTI-360 drive id forwarded to the JAX worker.",
    )
    parser.add_argument(
        "--use-service-labels",
        action="store_true",
        help=(
            "Find the running `vbogs-jax` sibling through Docker Compose labels. "
            "This is the normal mode from `vbogs-pipeline`."
        ),
    )
    parser.add_argument(
        "--label-project",
        default=os.environ.get("VBOGS_COMPOSE_PROJECT", ""),
        help=(
            "Compose project/Portainer stack label. Defaults to "
            "VBOGS_COMPOSE_PROJECT or auto-detects from this container."
        ),
    )
    parser.add_argument(
        "--jax-container",
        default="",
        help="Concrete running JAX container name/id. Overrides service-label lookup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the docker exec command without executing it.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "bucket", "stereo"),
        default="auto",
        help="Input mode forwarded to `train_vbgs_kitti_baseline.py`.",
    )
    parser.add_argument("--bucket-root", type=Path, default=None)
    parser.add_argument("--points-norm", type=Path, default=None)
    parser.add_argument("--norm-params", type=Path, default=None)
    parser.add_argument("--points-world", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional deterministic point cap for stereo input mode.",
    )
    parser.add_argument("--n-components", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reassign-fraction", type=float, default=0.05)
    parser.add_argument("--no-reassign", action="store_true")
    parser.add_argument("--project-anchors", action="store_true")
    parser.add_argument("--pts-by-anchor", type=Path, default=None)
    parser.add_argument("--eval-bucket-root", type=Path, default=None)
    parser.add_argument("--vbgs-root", type=Path, default=Path("vbgs"))
    return parser.parse_args(argv)


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
            f"found {len(matches)}. Pass --label-project or --jax-container."
        )
    return matches[0]


def docker_exec_prefix(args: argparse.Namespace) -> list[str]:
    if args.jax_container:
        container = args.jax_container
    elif args.use_service_labels:
        project = args.label_project or current_container_project()
        container = resolve_service_container(
            JAX_SERVICE,
            project=project,
            dry_run=args.dry_run,
        )
    else:
        raise RuntimeError(
            "Pass --use-service-labels from vbogs-pipeline, or pass "
            "--jax-container with a concrete running JAX container name/id."
        )
    return ["docker", "exec", "-i", "-w", "/workspace/VBOGS", container]


def maybe_option(flag: str, value: object | None) -> list[str]:
    if value is None or value == "":
        return []
    return [flag, str(value)]


def worker_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "python",
        "scripts/train_vbgs_kitti_baseline.py",
        "--drive",
        args.drive,
        "--input-mode",
        args.input_mode,
        "--max-points",
        str(args.max_points),
        "--n-components",
        str(args.n_components),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--reassign-fraction",
        str(args.reassign_fraction),
        "--vbgs-root",
        str(args.vbgs_root),
        *maybe_option("--bucket-root", args.bucket_root),
        *maybe_option("--points-norm", args.points_norm),
        *maybe_option("--norm-params", args.norm_params),
        *maybe_option("--points-world", args.points_world),
        *maybe_option("--output-root", args.output_root),
        *maybe_option("--pts-by-anchor", args.pts_by_anchor),
        *maybe_option("--eval-bucket-root", args.eval_bucket_root),
    ]
    if args.no_reassign:
        cmd.append("--no-reassign")
    if args.project_anchors:
        cmd.append("--project-anchors")
    return cmd


def build_exec_command(args: argparse.Namespace) -> list[str]:
    return [*docker_exec_prefix(args), *worker_command(args)]


def run_printed(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_printed(build_exec_command(args), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
