#!/usr/bin/env python3

"""List locally downloaded VBOGS dataset clips and pipeline selectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.dataset_inventory import clips_to_json, format_clip_table, list_dataset_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        choices=("all", "kitti360", "nvidia_ncore"),
        default="all",
        help="Dataset family to list.",
    )
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="Show only clips with the default files needed by the pipeline profile.",
    )
    parser.add_argument(
        "--commands",
        action="store_true",
        help="Also print the dataset selector arguments to pass to scripts/run_pipeline.sh.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table.",
    )
    parser.add_argument("--ncore-root", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None, help="KITTI-360 image root override.")
    parser.add_argument("--poses-root", type=Path, default=None, help="KITTI-360 poses root override.")
    parser.add_argument("--calibration-dir", type=Path, default=None, help="KITTI-360 calibration dir override.")
    parser.add_argument("--colmap-root", type=Path, default=None, help="Prepared COLMAP root override.")
    parser.add_argument("--octree-output-root", type=Path, default=None, help="Octree-AnyGS output root override.")
    parser.add_argument("--points-root", type=Path, default=None, help="World point-cloud artifact root override.")
    parser.add_argument("--bucket-root", type=Path, default=None, help="M4/M5 bucket artifact root override.")
    parser.add_argument("--outputs-root", type=Path, default=None, help="Curated/diagnostic outputs root override.")
    parser.add_argument("--m6-root", type=Path, default=None, help="M6 NBV artifact root override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = list_dataset_clips(
        dataset_name=args.dataset_name,
        ncore_root=args.ncore_root,
        raw_root=args.raw_root,
        poses_root=args.poses_root,
        calibration_dir=args.calibration_dir,
        colmap_root=args.colmap_root,
        octree_output_root=args.octree_output_root,
        points_root=args.points_root,
        bucket_root=args.bucket_root,
        outputs_root=args.outputs_root,
        m6_root=args.m6_root,
    )
    if args.ready_only:
        clips = [clip for clip in clips if clip.ready]

    if args.json:
        print(clips_to_json(clips))
    else:
        print(format_clip_table(clips, include_commands=args.commands))


if __name__ == "__main__":
    main()
