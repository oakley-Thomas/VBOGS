#!/usr/bin/env python3

"""Inspect an NVIDIA PhysicalAI AV NCore clip for experiment planning.

Lists the camera and LiDAR sensors available in a downloaded NCore clip along
with per-sensor frame counts, prints ready-to-copy `--camera-id` flag sets,
and suggests the largest `--max-frames` value that keeps the frame count
divisible by 8 (required by the experiment 04 fair-evaluation split).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.data_layout import resolve_nvidia_ncore_root
from vbogs.ncore_adapter import (
    DEFAULT_CAMERA_DEPTH_PAIR,
    get_frame_count,
    load_ncore_sequence_loader,
)

# Preferred variant order for experiment 04: front wide is always the primary
# (timestamp reference) camera, front tele is the first addition.
CAMERA_PRIORITY = DEFAULT_CAMERA_DEPTH_PAIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-id",
        required=True,
        help="NCore clip/sequence id under the NCore root.",
    )
    parser.add_argument(
        "--ncore-root",
        type=Path,
        default=None,
        help="Root containing converted NCore V4 clips. Defaults to repo/container data paths.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Frame step used to compute the suggested --max-frames value.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table.",
    )
    return parser.parse_args()


def inspect_clip_from_loader(loader: Any) -> dict[str, Any]:
    cameras: dict[str, dict[str, int]] = {}
    for camera_id in loader.camera_ids:
        sensor = loader.get_camera_sensor(camera_id)
        cameras[str(camera_id)] = {"frame_count": get_frame_count(sensor)}
    lidar_ids = [str(lidar_id) for lidar_id in getattr(loader, "lidar_ids", [])]
    return {
        "camera_ids": [str(camera_id) for camera_id in loader.camera_ids],
        "cameras": cameras,
        "lidar_ids": lidar_ids,
    }


def selectable_frames(frame_count: int, frame_step: int) -> int:
    if frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    return len(range(frame_count)[::frame_step])


def suggest_max_frames(frame_count: int, frame_step: int, multiple: int = 8) -> int:
    """Largest --max-frames value <= available frames that is a multiple of 8."""
    available = selectable_frames(frame_count, frame_step)
    return (available // multiple) * multiple


def ordered_camera_ids(camera_ids: list[str]) -> list[str]:
    """Deterministic priority order: known defaults first, remainder sorted."""
    preferred = [camera_id for camera_id in CAMERA_PRIORITY if camera_id in camera_ids]
    remainder = sorted(camera_id for camera_id in camera_ids if camera_id not in preferred)
    return preferred + remainder


def camera_flag_sets(camera_ids: list[str]) -> list[str]:
    ordered = ordered_camera_ids(camera_ids)
    flag_sets = []
    for count in range(1, len(ordered) + 1):
        flags = " ".join(f"--camera-id {camera_id}" for camera_id in ordered[:count])
        flag_sets.append(flags)
    return flag_sets


def main() -> None:
    args = parse_args()
    ncore_root = resolve_nvidia_ncore_root(args.ncore_root)
    loader = load_ncore_sequence_loader(ncore_root, args.scene_id)
    info = inspect_clip_from_loader(loader)

    if args.json:
        payload = dict(info)
        payload["scene_id"] = args.scene_id
        payload["ncore_root"] = str(ncore_root)
        payload["frame_step"] = args.frame_step
        payload["suggested_max_frames"] = {
            camera_id: suggest_max_frames(entry["frame_count"], args.frame_step)
            for camera_id, entry in info["cameras"].items()
        }
        payload["ordered_camera_ids"] = ordered_camera_ids(info["camera_ids"])
        print(json.dumps(payload, indent=2))
        return

    print(f"NCore clip     : {args.scene_id}")
    print(f"NCore root     : {ncore_root}")
    print()
    print("Cameras:")
    for camera_id in info["camera_ids"]:
        frame_count = info["cameras"][camera_id]["frame_count"]
        suggestion = suggest_max_frames(frame_count, args.frame_step)
        print(
            f"  {camera_id:<40s} frames={frame_count:<6d} "
            f"suggested --max-frames {suggestion} (step {args.frame_step})"
        )
    print()
    print("LiDARs:")
    for lidar_id in info["lidar_ids"]:
        print(f"  {lidar_id}")
    print()
    print("Camera flag sets (experiment 04 variants, primary camera first):")
    for count, flags in enumerate(camera_flag_sets(info["camera_ids"]), start=1):
        print(f"  cam{count}: {flags}")


if __name__ == "__main__":
    main()
