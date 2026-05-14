#!/usr/bin/env python3

"""Run exp02 pipeline profiles across KITTI-360 drives.

The heavy lifting stays in ``scripts/run_drive_pipeline.py``. This wrapper only
discovers drive ids, chooses the exp02 config profile(s), and launches the
single-drive runner once per drive/profile pair.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.data_layout import resolve_kitti360_path


DRIVE_GLOB = "2013_05_28_drive_*_sync"
EXP02_CONFIGS = {
    "explicit": REPO_ROOT / "outputs/experiments/exp02-A_explicit3d_baseline_pipeline_config.yaml",
    "implicit": REPO_ROOT / "outputs/experiments/exp02_B_implicit3d_baseline_pipeline_config.yaml",
}


def fs_path(path: Path) -> Path:
    """Return the local filesystem path used by this wrapper for inspection."""

    return path.expanduser() if path.is_absolute() else (Path.cwd() / path).resolve()


def default_input_root(kind: str) -> Path:
    """Resolve repo-local KITTI-360 defaults from the wrapper's repo root."""

    path = resolve_kitti360_path(None, kind=kind)
    return path if path.is_absolute() else REPO_ROOT / path


def drive_has_stereo(raw_root: Path, drive: str) -> bool:
    drive_root = raw_root / drive
    return (
        (drive_root / "image_00" / "data_rect").is_dir()
        and (drive_root / "image_01" / "data_rect").is_dir()
    )


def drive_has_pose(poses_root: Path, drive: str) -> bool:
    return (poses_root / drive / "cam0_to_world.txt").is_file()


def discover_drives(raw_root: Path, poses_root: Path) -> list[str]:
    """Find KITTI-360 drives that have both stereo images and camera poses."""

    if not raw_root.is_dir():
        raise FileNotFoundError(f"KITTI-360 raw image root not found: {raw_root}")
    if not poses_root.is_dir():
        raise FileNotFoundError(f"KITTI-360 poses root not found: {poses_root}")

    drives: list[str] = []
    for path in sorted(raw_root.glob(DRIVE_GLOB)):
        if not path.is_dir():
            continue
        drive = path.name
        if drive_has_stereo(raw_root, drive) and drive_has_pose(poses_root, drive):
            drives.append(drive)
    return drives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        choices=("implicit", "explicit", "both"),
        default="implicit",
        help=(
            "Exp02 profile to run. The default matches exp02_B, the currently "
            "active implicit3D baseline; use 'both' for A+B."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=None,
        help=(
            "Pipeline config to run instead of --variant. May be repeated. "
            "Each config is passed to scripts/run_drive_pipeline.py."
        ),
    )
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        default=None,
        help=(
            "Drive id to run. May be repeated. If omitted, drives are discovered "
            "from the KITTI-360 image and pose roots."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="KITTI-360 raw image root used for discovery and forwarded to the pipeline.",
    )
    parser.add_argument(
        "--poses-root",
        type=Path,
        default=None,
        help="KITTI-360 poses root used for discovery and forwarded to the pipeline.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="KITTI-360 calibration directory forwarded to the pipeline.",
    )
    parser.add_argument(
        "--list-drives",
        action="store_true",
        help="Print discovered/selected drives and exit without launching pipelines.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-drive pipeline commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later drive/profile pairs if one run fails.",
    )
    return parser


def config_paths(args: argparse.Namespace) -> list[Path]:
    if args.config:
        return [path if path.is_absolute() else (REPO_ROOT / path) for path in args.config]
    if args.variant == "both":
        return [EXP02_CONFIGS["explicit"], EXP02_CONFIGS["implicit"]]
    return [EXP02_CONFIGS[args.variant]]


def selected_drives(args: argparse.Namespace) -> list[str]:
    if args.drives:
        return sorted(dict.fromkeys(args.drives))

    raw_root = fs_path(args.raw_root) if args.raw_root is not None else default_input_root("raw")
    poses_root = (
        fs_path(args.poses_root) if args.poses_root is not None else default_input_root("poses")
    )
    drives = discover_drives(raw_root, poses_root)
    if not drives:
        raise RuntimeError(
            "No KITTI-360 drives with image_00/image_01 stereo data and cam0 poses "
            f"were found under raw_root={raw_root} poses_root={poses_root}"
        )
    return drives


def path_for_command(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_pipeline_command(
    *,
    config: Path,
    drive: str,
    args: argparse.Namespace,
    extra_args: Sequence[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_drive_pipeline.py",
        "--config",
        path_for_command(config),
        "--drive",
        drive,
    ]
    for value, flag in (
        (args.raw_root, "--raw-root"),
        (args.poses_root, "--poses-root"),
        (args.calibration_dir, "--calibration-dir"),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    cmd.extend(extra_args)
    return cmd


def normalize_extra_args(extra_args: Sequence[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return list(extra_args[1:])
    return list(extra_args)


def run_printed(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def print_drives(drives: Sequence[str]) -> None:
    for drive in drives:
        print(drive)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)
    extra_args = normalize_extra_args(extra_args)
    configs = config_paths(args)
    drives = selected_drives(args)

    if args.list_drives:
        print_drives(drives)
        return

    print("Exp02 configs:")
    for config in configs:
        print(f"  - {path_for_command(config)}")
    print("Drives:")
    for drive in drives:
        print(f"  - {drive}")

    failures: list[tuple[str, Path, int]] = []
    for drive in drives:
        for config in configs:
            print(f"\n=== {drive} :: {config.stem} ===", flush=True)
            cmd = build_pipeline_command(
                config=config,
                drive=drive,
                args=args,
                extra_args=extra_args,
            )
            try:
                run_printed(cmd, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                if not args.continue_on_error:
                    raise
                failures.append((drive, config, exc.returncode))
                print(
                    f"Run failed with exit code {exc.returncode}; continuing.",
                    file=sys.stderr,
                    flush=True,
                )

    if failures:
        print("\nFailed exp02 runs:", file=sys.stderr)
        for drive, config, returncode in failures:
            print(
                f"  - {drive} :: {path_for_command(config)} exited {returncode}",
                file=sys.stderr,
            )
        sys.exit(1)

    print("\nExp02 drive sweep completed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
