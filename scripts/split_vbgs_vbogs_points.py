#!/usr/bin/env python3

"""Split KITTI-360 stereo point artifacts into train/eval frame subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json


DEFAULT_POINTS_ROOT = REPO_ROOT / "data" / "points_world"
DEFAULT_COLMAP_ROOT = Path("/data/COLMAP")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vbgs_comparison"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0007_sync",
        help="KITTI-360 drive id.",
    )
    parser.add_argument(
        "--points-world",
        type=Path,
        default=None,
        help="Input `points_world.npz`. Defaults to `data/points_world/<drive>/points_world.npz`.",
    )
    parser.add_argument(
        "--selection-metadata",
        type=Path,
        default=None,
        help="Prepared COLMAP metadata. Defaults to `/data/COLMAP/<drive>/metadata.json`.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to `outputs/vbgs_comparison/<drive>`.",
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="Every Nth selected frame is assigned to eval.",
    )
    return parser.parse_args(argv)


def resolve_points_world_path(drive: str, points_world: Path | None) -> Path:
    if points_world is not None:
        return points_world.resolve()
    return (DEFAULT_POINTS_ROOT / drive / "points_world.npz").resolve()


def resolve_selection_metadata(drive: str, selection_metadata: Path | None) -> Path:
    if selection_metadata is not None:
        return selection_metadata.resolve()
    return (DEFAULT_COLMAP_ROOT / drive / "metadata.json").resolve()


def resolve_output_root(drive: str, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.resolve()
    return (DEFAULT_OUTPUT_ROOT / drive).resolve()


def load_selected_frames(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    selected = metadata.get("selected_frames")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"`selected_frames` missing or empty in {path}")
    return [int(frame_id) for frame_id in selected]


def split_selected_frames(
    selected_frames: Sequence[int],
    *,
    llffhold: int,
) -> tuple[list[int], list[int]]:
    if llffhold <= 0:
        raise ValueError("--llffhold must be positive")
    eval_frames = [
        int(frame_id)
        for idx, frame_id in enumerate(selected_frames)
        if idx % llffhold == 0
    ]
    train_frames = [
        int(frame_id)
        for idx, frame_id in enumerate(selected_frames)
        if idx % llffhold != 0
    ]
    if not train_frames:
        raise ValueError("Frame split produced no train frames")
    if not eval_frames:
        raise ValueError("Frame split produced no eval frames")
    return train_frames, eval_frames


def subset_points(
    *,
    xyz: np.ndarray,
    rgb: np.ndarray,
    frame_id: np.ndarray,
    frames: Sequence[int],
) -> dict[str, np.ndarray]:
    frame_set = np.asarray(list(frames), dtype=np.int32)
    mask = np.isin(frame_id, frame_set)
    return {
        "xyz": xyz[mask].astype(np.float32, copy=False),
        "rgb": rgb[mask].astype(np.uint8, copy=False),
        "frame_id": frame_id[mask].astype(np.int32, copy=False),
    }


def save_points(path: Path, subset: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=subset["xyz"],
        rgb=subset["rgb"],
        frame_id=subset["frame_id"],
    )


def split_points(
    *,
    drive: str,
    points_world_path: Path,
    selection_metadata_path: Path,
    output_root: Path,
    llffhold: int,
) -> dict[str, Any]:
    selected_frames = load_selected_frames(selection_metadata_path)
    train_frames, eval_frames = split_selected_frames(
        selected_frames,
        llffhold=llffhold,
    )

    with np.load(points_world_path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = np.asarray(data["rgb"], dtype=np.uint8)
        frame_id = np.asarray(data["frame_id"], dtype=np.int32)

    train_subset = subset_points(
        xyz=xyz,
        rgb=rgb,
        frame_id=frame_id,
        frames=train_frames,
    )
    eval_subset = subset_points(
        xyz=xyz,
        rgb=rgb,
        frame_id=frame_id,
        frames=eval_frames,
    )
    if train_subset["xyz"].shape[0] == 0:
        raise ValueError("Point split produced no train points")
    if eval_subset["xyz"].shape[0] == 0:
        raise ValueError("Point split produced no eval points")

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "points_train.npz"
    eval_path = output_root / "points_eval.npz"
    metadata_path = output_root / "split_metadata.json"
    save_points(train_path, train_subset)
    save_points(eval_path, eval_subset)

    metadata = {
        "drive": drive,
        "points_world_path": str(points_world_path),
        "selection_metadata_path": str(selection_metadata_path),
        "llffhold": int(llffhold),
        "selected_frame_count": len(selected_frames),
        "train_frames": train_frames,
        "eval_frames": eval_frames,
        "train_frame_count": len(train_frames),
        "eval_frame_count": len(eval_frames),
        "source_point_count": int(xyz.shape[0]),
        "train_point_count": int(train_subset["xyz"].shape[0]),
        "eval_point_count": int(eval_subset["xyz"].shape[0]),
        "points_train": str(train_path),
        "points_eval": str(eval_path),
    }
    save_json(metadata_path, metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = resolve_output_root(args.drive, args.output_root)
    metadata = split_points(
        drive=args.drive,
        points_world_path=resolve_points_world_path(args.drive, args.points_world),
        selection_metadata_path=resolve_selection_metadata(
            args.drive,
            args.selection_metadata,
        ),
        output_root=output_root,
        llffhold=args.llffhold,
    )
    print(f"Wrote {metadata['points_train']}")
    print(f"Wrote {metadata['points_eval']}")
    print(f"Wrote {output_root / 'split_metadata.json'}")


if __name__ == "__main__":
    main()
