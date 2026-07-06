"""Inventory helpers for downloaded VBOGS source datasets."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from vbogs.data_layout import resolve_kitti360_path, resolve_nvidia_ncore_root


NCORE_DEFAULT_CAMERA_IDS = (
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
)
NCORE_DEFAULT_LIDAR_ID = "lidar_top_360fov"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetClip:
    dataset: str
    scene_id: str
    status: str
    path: str
    files: dict[str, int | bool | str]
    notes: tuple[str, ...]
    pipeline_args: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def _count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        1
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _status_from_notes(notes: list[str]) -> str:
    return "ready" if not notes else "partial"


def list_kitti360_clips(
    *,
    raw_root: Path | None = None,
    poses_root: Path | None = None,
    calibration_dir: Path | None = None,
) -> list[DatasetClip]:
    """List locally downloaded KITTI-360 drives."""

    raw_root = resolve_kitti360_path(raw_root, kind="raw")
    poses_root = resolve_kitti360_path(poses_root, kind="poses")
    calibration_dir = resolve_kitti360_path(calibration_dir, kind="calibration")
    perspective_path = calibration_dir / "perspective.txt"

    scene_ids: set[str] = set()
    if raw_root.is_dir():
        scene_ids.update(item.name for item in raw_root.iterdir() if item.is_dir())
    if poses_root.is_dir():
        scene_ids.update(item.name for item in poses_root.iterdir() if item.is_dir())

    clips: list[DatasetClip] = []
    for scene_id in sorted(scene_ids):
        scene_root = raw_root / scene_id
        left_dir = scene_root / "image_00" / "data_rect"
        right_dir = scene_root / "image_01" / "data_rect"
        pose_path = poses_root / scene_id / "cam0_to_world.txt"
        left_images = _count_images(left_dir)
        right_images = _count_images(right_dir)
        notes: list[str] = []
        if left_images == 0:
            notes.append("missing left rectified images")
        if right_images == 0:
            notes.append("missing right rectified images")
        if not pose_path.exists():
            notes.append("missing poses cam0_to_world.txt")
        if not perspective_path.exists():
            notes.append("missing calibration perspective.txt")

        clips.append(
            DatasetClip(
                dataset="kitti360",
                scene_id=scene_id,
                status=_status_from_notes(notes),
                path=str(scene_root),
                files={
                    "left_images": left_images,
                    "right_images": right_images,
                    "poses": pose_path.exists(),
                    "calibration": perspective_path.exists(),
                },
                notes=tuple(notes),
                pipeline_args=("--drive", scene_id),
            )
        )
    return clips


def _scene_id_from_ncore_dir(path: Path) -> str:
    name = path.name
    if name.startswith("pai_"):
        return name[4:]
    return name


def _scene_id_from_ncore_file(path: Path) -> str | None:
    name = path.name
    match = re.match(r"^pai_(?P<scene>.+?)\.ncore4(?:-.+)?\.zarr\.itar$", name)
    if match:
        return match.group("scene")
    match = re.match(r"^(?P<scene>.+?)\.ncore4(?:-.+)?\.zarr\.itar$", name)
    if match:
        return match.group("scene")
    match = re.match(r"^pai_(?P<scene>.+?)\.json$", name)
    if match:
        return match.group("scene")
    return None


def _ncore_scene_paths(root: Path, scene_id: str, scene_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    if scene_dir is not None and scene_dir.is_dir():
        paths.extend(
            sorted(
                item
                for item in scene_dir.iterdir()
                if item.is_file()
                and (
                    item.suffix == ".json"
                    or item.name.endswith(".ncore4.zarr.itar")
                    or ".ncore4-" in item.name
                    or item.name == "sequence-ncore4.json"
                )
            )
        )
    if root.is_dir():
        for item in root.iterdir():
            if not item.is_file():
                continue
            if _scene_id_from_ncore_file(item) == scene_id:
                paths.append(item)
    return sorted(set(paths))


def _ncore_component_summary(paths: Sequence[Path], scene_id: str) -> tuple[dict[str, int | bool | str], list[str]]:
    names = [path.name for path in paths]
    metadata = any(name in {f"pai_{scene_id}.json", f"{scene_id}.json", "sequence-ncore4.json"} for name in names)
    core = any(
        name in {f"pai_{scene_id}.ncore4.zarr.itar", f"{scene_id}.ncore4.zarr.itar", "sequence-ncore4.json"}
        for name in names
    )
    camera_components = sorted(
        name for name in names if f"pai_{scene_id}.ncore4-camera_" in name or ".ncore4-camera_" in name
    )
    lidar_components = sorted(
        name for name in names if f"pai_{scene_id}.ncore4-lidar_" in name or ".ncore4-lidar_" in name
    )

    notes: list[str] = []
    if not core:
        notes.append("missing core NCore sequence component")
    for camera_id in NCORE_DEFAULT_CAMERA_IDS:
        if not any(camera_id in name for name in camera_components):
            notes.append(f"missing default camera component {camera_id}")
    if not any(NCORE_DEFAULT_LIDAR_ID in name for name in lidar_components):
        notes.append(f"missing default lidar component {NCORE_DEFAULT_LIDAR_ID}")
    if not metadata:
        notes.append("missing metadata json")

    files: dict[str, int | bool | str] = {
        "components": len(paths),
        "metadata": metadata,
        "core": core,
        "camera_components": len(camera_components),
        "lidar_components": len(lidar_components),
    }
    return files, notes


def list_nvidia_ncore_clips(ncore_root: Path | None = None) -> list[DatasetClip]:
    """List locally downloaded NVIDIA PhysicalAI AV NCore clips."""

    root = resolve_nvidia_ncore_root(ncore_root)
    if not root.is_dir():
        return []

    scene_dirs: dict[str, Path | None] = {}
    for item in root.iterdir():
        if item.is_dir():
            scene_id = _scene_id_from_ncore_dir(item)
            paths = _ncore_scene_paths(root, scene_id, item)
            if paths:
                scene_dirs[scene_id] = item
        elif item.is_file():
            scene_id = _scene_id_from_ncore_file(item)
            if scene_id:
                scene_dirs.setdefault(scene_id, None)

    clips: list[DatasetClip] = []
    for scene_id, scene_dir in sorted(scene_dirs.items()):
        paths = _ncore_scene_paths(root, scene_id, scene_dir)
        files, notes = _ncore_component_summary(paths, scene_id)
        clips.append(
            DatasetClip(
                dataset="nvidia_ncore",
                scene_id=scene_id,
                status=_status_from_notes(notes),
                path=str(scene_dir or root),
                files=files,
                notes=tuple(notes),
                pipeline_args=(
                    "--config",
                    "configs/pipeline/nvidia_ncore_dev.yaml",
                    "--dataset-name",
                    "nvidia_ncore",
                    "--scene-id",
                    scene_id,
                ),
            )
        )
    return clips


def list_dataset_clips(
    *,
    dataset_name: str = "all",
    ncore_root: Path | None = None,
    raw_root: Path | None = None,
    poses_root: Path | None = None,
    calibration_dir: Path | None = None,
) -> list[DatasetClip]:
    clips: list[DatasetClip] = []
    if dataset_name in {"all", "kitti360"}:
        clips.extend(
            list_kitti360_clips(
                raw_root=raw_root,
                poses_root=poses_root,
                calibration_dir=calibration_dir,
            )
        )
    if dataset_name in {"all", "nvidia_ncore"}:
        clips.extend(list_nvidia_ncore_clips(ncore_root))
    return sorted(clips, key=lambda clip: (clip.dataset, clip.scene_id))


def clips_to_json(clips: Iterable[DatasetClip]) -> str:
    payload = []
    for clip in clips:
        record = asdict(clip)
        record["ready"] = clip.ready
        payload.append(record)
    return json.dumps(payload, indent=2, sort_keys=True)


def format_clip_table(clips: Sequence[DatasetClip], *, include_commands: bool = False) -> str:
    if not clips:
        return "No downloaded dataset clips found."

    rows = [
        (
            clip.dataset,
            clip.scene_id,
            clip.status,
            _file_summary(clip),
            "; ".join(clip.notes) if clip.notes else "ok",
        )
        for clip in clips
    ]
    headers = ("dataset", "scene_id", "status", "files", "notes")
    widths = [
        max(len(str(row[index])) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(str(headers[index]).ljust(widths[index]) for index in range(len(headers))),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    for row in rows:
        lines.append("  ".join(str(row[index]).ljust(widths[index]) for index in range(len(headers))))

    if include_commands:
        lines.append("")
        lines.append("Pipeline selectors:")
        for clip in clips:
            lines.append(f"  {clip.scene_id}: {' '.join(clip.pipeline_args)}")
    return "\n".join(lines)


def _file_summary(clip: DatasetClip) -> str:
    if clip.dataset == "kitti360":
        return f"L={clip.files['left_images']} R={clip.files['right_images']}"
    if clip.dataset == "nvidia_ncore":
        return (
            f"components={clip.files['components']} "
            f"cameras={clip.files['camera_components']} "
            f"lidars={clip.files['lidar_components']}"
        )
    return ""
