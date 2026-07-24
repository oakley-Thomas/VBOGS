"""Dataset split and metadata camera helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from vbogs.viewer.camera import ViewerCamera, _viewer_resolution


SPLIT_NAMES = ("train", "test", "validation")
DEFAULT_SPLIT_RATIOS = {
    "train": 0.7,
    "test": 0.2,
    "validation": 0.1,
}


def split_counts(total_count: int) -> dict[str, int]:
    """Return deterministic 70/20/10 counts that sum to ``total_count``."""

    total_count = int(total_count)
    if total_count < 0:
        raise ValueError("total_count must be non-negative")
    if total_count == 0:
        return {name: 0 for name in SPLIT_NAMES}

    raw = {name: DEFAULT_SPLIT_RATIOS[name] * total_count for name in SPLIT_NAMES}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total_count - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw[name] - counts[name], -DEFAULT_SPLIT_RATIOS[name]),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    if counts["train"] == 0:
        donor = max(("test", "validation"), key=lambda name: counts[name])
        if counts[donor] > 0:
            counts[donor] -= 1
            counts["train"] = 1
    return counts


def _pick_uniform_indices(
    available: list[int],
    count: int,
) -> list[int]:
    if count <= 0:
        return []
    if count > len(available):
        raise ValueError("Cannot pick more split entries than available frames")

    chosen: list[int] = []
    targets = (np.arange(count, dtype=np.float64) + 0.5) * len(available) / count - 0.5
    for target in targets:
        ranked = sorted(
            range(len(available)),
            key=lambda idx: (abs(idx - target), -idx),
        )
        for idx in ranked:
            value = available[idx]
            if value not in chosen:
                chosen.append(value)
                break
    return sorted(chosen)


def split_frame_indices(selected_frames: Sequence[int]) -> dict[str, list[int]]:
    """Assign selected frame ids to deterministic timeline-uniform splits."""

    frames = [int(frame_id) for frame_id in selected_frames]
    counts = split_counts(len(frames))
    available_positions = list(range(len(frames)))

    split_positions: dict[str, list[int]] = {}
    for name in ("validation", "test"):
        positions = _pick_uniform_indices(available_positions, counts[name])
        split_positions[name] = positions
        position_set = set(positions)
        available_positions = [pos for pos in available_positions if pos not in position_set]

    split_positions["train"] = available_positions
    return {
        name: [frames[pos] for pos in sorted(split_positions[name])]
        for name in SPLIT_NAMES
    }


def split_lookup(frame_splits: dict[str, Sequence[int]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for split_name, frames in frame_splits.items():
        for frame_id in frames:
            lookup[int(frame_id)] = str(split_name)
    return lookup


def load_split_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def records_for_split(metadata: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split == "all":
        wanted = None
    elif split in SPLIT_NAMES:
        wanted = split
    else:
        raise ValueError(f"Unsupported split: {split}")

    records = metadata.get("frame_records")
    if not isinstance(records, list):
        return []
    result = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_split = record.get("split")
        if wanted is None or record_split == wanted:
            result.append(record)
    return result


def frames_for_split(metadata: dict[str, Any], split: str) -> list[int]:
    frame_splits = metadata.get("frame_splits")
    if split == "all":
        selected = metadata.get("selected_frames", [])
        return [int(frame_id) for frame_id in selected]
    if isinstance(frame_splits, dict) and split in frame_splits:
        return [int(frame_id) for frame_id in frame_splits[split]]
    return [int(record["frame_id"]) for record in records_for_split(metadata, split)]


def _fallback_source(source_cam: Any | None) -> Any:
    return source_cam or SimpleNamespace(
        resolution_scale=1.0,
        znear=0.01,
        zfar=100.0,
        image_path="",
    )


def _camera_from_payload(
    *,
    payload: dict[str, Any],
    intrinsics: dict[str, Any],
    source_cam: Any,
    uid: int,
    resolution_arg: int | float,
    device: str,
) -> ViewerCamera:
    orig_w = int(intrinsics["width"])
    orig_h = int(intrinsics["height"])
    width, height = _viewer_resolution(orig_w, orig_h, 1.0, resolution_arg)
    sx = float(width) / float(orig_w)
    sy = float(height) / float(orig_h)
    image_name = str(payload["image_name"])
    return ViewerCamera(
        source_cam=source_cam,
        c2w_np=np.asarray(payload["c2w"], dtype=np.float32),
        uid=uid,
        image_name=Path(image_name).stem,
        width=width,
        height=height,
        fx=float(intrinsics["fx"]) * sx,
        fy=float(intrinsics["fy"]) * sy,
        cx=float(intrinsics["cx"]) * sx,
        cy=float(intrinsics["cy"]) * sy,
        znear=float(getattr(source_cam, "znear", 0.01)),
        zfar=float(getattr(source_cam, "zfar", 100.0)),
        device=device,
    )


def _ncore_metadata_cameras(
    metadata: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    source_cam: Any,
    resolution_arg: int | float,
    device: str,
) -> list[ViewerCamera]:
    intrinsics = metadata.get("intrinsics", {})
    if not isinstance(intrinsics, dict):
        return []

    cameras: list[ViewerCamera] = []
    uid = 0
    for record in records:
        camera_payloads = record.get("cameras", {})
        if not isinstance(camera_payloads, dict):
            continue
        for camera_id in metadata.get("camera_ids", sorted(camera_payloads)):
            if camera_id not in camera_payloads or camera_id not in intrinsics:
                continue
            cameras.append(
                _camera_from_payload(
                    payload=camera_payloads[camera_id],
                    intrinsics=intrinsics[camera_id],
                    source_cam=source_cam,
                    uid=uid,
                    resolution_arg=resolution_arg,
                    device=device,
                )
            )
            uid += 1
    return cameras


def _kitti_metadata_cameras(
    metadata: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    source_cam: Any,
    resolution_arg: int | float,
    device: str,
) -> list[ViewerCamera]:
    intrinsics = metadata.get("camera_intrinsics", {})
    if not isinstance(intrinsics, dict):
        return []

    cameras: list[ViewerCamera] = []
    uid = 0
    for record in records:
        images = record.get("images", [])
        if not isinstance(images, list):
            continue
        for image_payload in images:
            if not isinstance(image_payload, dict):
                continue
            camera_label = image_payload.get("camera")
            if camera_label not in intrinsics:
                continue
            cameras.append(
                _camera_from_payload(
                    payload=image_payload,
                    intrinsics=intrinsics[camera_label],
                    source_cam=source_cam,
                    uid=uid,
                    resolution_arg=resolution_arg,
                    device=device,
                )
            )
            uid += 1
    return cameras


def metadata_cameras_for_split(
    metadata: dict[str, Any],
    split: str,
    *,
    resolution_arg: int | float,
    source_cam: Any | None = None,
    device: str = "cuda",
) -> list[ViewerCamera]:
    """Build render-compatible cameras from prepared dataset metadata."""

    records = records_for_split(metadata, split)
    if not records:
        return []

    source = _fallback_source(source_cam)
    if metadata.get("dataset") == "nvidia_ncore":
        return _ncore_metadata_cameras(
            metadata,
            records,
            source_cam=source,
            resolution_arg=resolution_arg,
            device=device,
        )
    return _kitti_metadata_cameras(
        metadata,
        records,
        source_cam=source,
        resolution_arg=resolution_arg,
        device=device,
    )
