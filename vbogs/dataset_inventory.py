"""Inventory helpers for downloaded VBOGS source datasets and artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from vbogs.data_layout import resolve_kitti360_path, resolve_nvidia_ncore_root


NCORE_DEFAULT_CAMERA_IDS = (
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
)
NCORE_DEFAULT_LIDAR_ID = "lidar_top_360fov"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PIPELINE_STAGES = (
    "prepare",
    "train",
    "stereo",
    "bucket",
    "fit",
    "uncertainty",
    "map-viz",
    "render",
    "nbv",
    "nbv-viz",
    "bundle",
)
KITTI360_DRIVE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_drive_\d{4}_sync$")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass(frozen=True)
class DatasetClip:
    dataset: str
    scene_id: str
    status: str
    path: str
    files: dict[str, int | bool | str]
    notes: tuple[str, ...]
    pipeline_args: tuple[str, ...]
    trained: bool = False
    latest_stage: str | None = None
    stage_outputs: dict[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def stage_output_paths(self) -> dict[str, str]:
        return self.stage_outputs


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


def _existing_roots(explicit_root: Path | None, defaults: Sequence[Path]) -> list[Path]:
    if explicit_root is not None:
        return [explicit_root]
    return list(defaults)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_trained_run(scene_root: Path) -> Path | None:
    if not scene_root.is_dir():
        return None
    candidates = sorted(path for path in scene_root.iterdir() if path.is_dir())
    for run_dir in reversed(candidates):
        if not (run_dir / "config.yaml").is_file():
            continue
        point_cloud_root = run_dir / "point_cloud"
        if not point_cloud_root.is_dir():
            continue
        if any(
            iteration_dir.is_dir() and (iteration_dir / "point_cloud_anchor.ply").is_file()
            for iteration_dir in point_cloud_root.iterdir()
        ):
            return run_dir
    return None


def _pipeline_args_for_dataset(dataset: str, scene_id: str) -> tuple[str, ...]:
    if dataset == "nvidia_ncore":
        return (
            "--config",
            "configs/pipeline/nvidia_ncore_dev.yaml",
            "--dataset-name",
            "nvidia_ncore",
            "--scene-id",
            scene_id,
        )
    if dataset == "kitti360":
        return ("--drive", scene_id)
    return ("--scene-id", scene_id)


def _infer_dataset(scene_id: str, stage_outputs: dict[str, str]) -> str:
    for stage in ("prepare", "stereo", "bundle"):
        path_text = stage_outputs.get(stage)
        if not path_text:
            continue
        path = Path(path_text)
        if stage == "prepare":
            metadata = _read_json(path)
        elif stage == "stereo":
            metadata = _read_json(path.with_name("points_world_metadata.json"))
        else:
            metadata = _read_json(path)
        dataset = metadata.get("dataset")
        if dataset in {"kitti360", "nvidia_ncore"}:
            return dataset
        points = metadata.get("points")
        if (
            isinstance(points, dict)
            and points.get("dataset") in {"kitti360", "nvidia_ncore"}
        ):
            return str(points["dataset"])
    if KITTI360_DRIVE_RE.match(scene_id):
        return "kitti360"
    if UUID_RE.match(scene_id):
        return "nvidia_ncore"
    return "artifact"


def _discover_stage_outputs(
    scene_id: str,
    *,
    colmap_roots: Sequence[Path],
    octree_output_roots: Sequence[Path],
    points_root: Path,
    bucket_root: Path,
    outputs_root: Path,
    m6_root: Path,
) -> dict[str, str]:
    stage_outputs: dict[str, str] = {}

    prepared_metadata = _first_existing(
        root / scene_id / "metadata.json" for root in colmap_roots
    )
    if prepared_metadata is not None:
        stage_outputs["prepare"] = str(prepared_metadata)

    for root in octree_output_roots:
        run_dir = _latest_trained_run(root / scene_id)
        if run_dir is not None:
            stage_outputs["train"] = str(run_dir)
            break

    points_npz = points_root / scene_id / "points_world.npz"
    points_metadata = points_root / scene_id / "points_world_metadata.json"
    if points_npz.is_file() and points_metadata.is_file():
        stage_outputs["stereo"] = str(points_npz)

    scene_bucket_root = bucket_root / scene_id
    if all(
        (scene_bucket_root / name).is_file()
        for name in (
            "points_norm.npz",
            "pts_by_anchor.npz",
            "norm_params.json",
            "bucket_metadata.json",
        )
    ):
        stage_outputs["bucket"] = str(scene_bucket_root / "bucket_metadata.json")

    if all(
        (scene_bucket_root / name).is_file()
        for name in ("anchor_posterior.npz", "fit_metadata.json")
    ):
        stage_outputs["fit"] = str(scene_bucket_root / "anchor_posterior.npz")

    if all(
        (scene_bucket_root / name).is_file()
        for name in ("U.npy", "uncertainty_components.npz", "uncertainty_metadata.json")
    ):
        stage_outputs["uncertainty"] = str(scene_bucket_root / "U.npy")

    map_dir = outputs_root / "uncertainty_maps" / scene_id
    if (
        (map_dir / "uncertainty_map_metadata.json").is_file()
        and (map_dir / "anchors_uncertainty_all.ply").is_file()
    ):
        stage_outputs["map-viz"] = str(map_dir / "uncertainty_map_metadata.json")

    render_dir = outputs_root / "uncertainty_views" / scene_id
    if (render_dir / "metadata.json").is_file():
        stage_outputs["render"] = str(render_dir / "metadata.json")

    nbv_path = _first_existing(
        (
            m6_root / scene_id / "nbv_scores.json",
            outputs_root / "v1_0" / scene_id / "nbv" / "nbv_scores.json",
        )
    )
    if nbv_path is not None:
        stage_outputs["nbv"] = str(nbv_path)

    nbv_viz_path = _first_existing(
        (
            m6_root / scene_id / "viz" / "viz_summary.json",
            outputs_root / "v1_0" / scene_id / "nbv" / "viz" / "viz_summary.json",
        )
    )
    if nbv_viz_path is not None:
        stage_outputs["nbv-viz"] = str(nbv_viz_path)

    bundle_manifest = outputs_root / "v1_0" / scene_id / "run_manifest.json"
    if bundle_manifest.is_file():
        stage_outputs["bundle"] = str(bundle_manifest)

    return stage_outputs


def _latest_stage(stage_outputs: dict[str, str]) -> str | None:
    latest = None
    for stage in PIPELINE_STAGES:
        if stage in stage_outputs:
            latest = stage
    return latest


def _artifact_scene_ids(
    *,
    colmap_roots: Sequence[Path],
    octree_output_roots: Sequence[Path],
    points_root: Path,
    bucket_root: Path,
    outputs_root: Path,
    m6_root: Path,
) -> set[str]:
    scene_ids: set[str] = set()
    for root in (*colmap_roots, *octree_output_roots, points_root, bucket_root):
        if root.is_dir():
            scene_ids.update(item.name for item in root.iterdir() if item.is_dir())
    for root in (
        outputs_root / "uncertainty_maps",
        outputs_root / "uncertainty_views",
        outputs_root / "v1_0",
        m6_root,
    ):
        if root.is_dir():
            scene_ids.update(item.name for item in root.iterdir() if item.is_dir())
    return scene_ids


def _with_artifact_status(clip: DatasetClip, stage_outputs: dict[str, str]) -> DatasetClip:
    return DatasetClip(
        dataset=clip.dataset,
        scene_id=clip.scene_id,
        status=clip.status,
        path=clip.path,
        files=clip.files,
        notes=clip.notes,
        pipeline_args=clip.pipeline_args,
        trained="train" in stage_outputs,
        latest_stage=_latest_stage(stage_outputs),
        stage_outputs=dict(stage_outputs),
    )


def _artifact_only_clip(scene_id: str, stage_outputs: dict[str, str]) -> DatasetClip:
    dataset = _infer_dataset(scene_id, stage_outputs)
    path = next(iter(stage_outputs.values()), "")
    return DatasetClip(
        dataset=dataset,
        scene_id=scene_id,
        status="artifact-only",
        path=path,
        files={},
        notes=("source dataset not found",),
        pipeline_args=_pipeline_args_for_dataset(dataset, scene_id),
        trained="train" in stage_outputs,
        latest_stage=_latest_stage(stage_outputs),
        stage_outputs=dict(stage_outputs),
    )


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
    colmap_root: Path | None = None,
    octree_output_root: Path | None = None,
    points_root: Path | None = None,
    bucket_root: Path | None = None,
    outputs_root: Path | None = None,
    m6_root: Path | None = None,
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

    colmap_roots = _existing_roots(colmap_root, (Path("/data/COLMAP"), Path("data/COLMAP")))
    octree_output_roots = _existing_roots(
        octree_output_root,
        (Path("/data/OCTREE-ANYGS"), Path("data/OCTREE-ANYGS")),
    )
    resolved_points_root = points_root or Path("data/points_world")
    resolved_bucket_root = bucket_root or Path("data/m4")
    resolved_outputs_root = outputs_root or Path("outputs")
    resolved_m6_root = m6_root or Path("data/m6")

    clip_by_scene = {clip.scene_id: clip for clip in clips}
    artifact_scene_ids = _artifact_scene_ids(
        colmap_roots=colmap_roots,
        octree_output_roots=octree_output_roots,
        points_root=resolved_points_root,
        bucket_root=resolved_bucket_root,
        outputs_root=resolved_outputs_root,
        m6_root=resolved_m6_root,
    )

    merged: list[DatasetClip] = []
    for clip in clips:
        stage_outputs = _discover_stage_outputs(
            clip.scene_id,
            colmap_roots=colmap_roots,
            octree_output_roots=octree_output_roots,
            points_root=resolved_points_root,
            bucket_root=resolved_bucket_root,
            outputs_root=resolved_outputs_root,
            m6_root=resolved_m6_root,
        )
        merged.append(_with_artifact_status(clip, stage_outputs))

    for scene_id in sorted(artifact_scene_ids - set(clip_by_scene)):
        stage_outputs = _discover_stage_outputs(
            scene_id,
            colmap_roots=colmap_roots,
            octree_output_roots=octree_output_roots,
            points_root=resolved_points_root,
            bucket_root=resolved_bucket_root,
            outputs_root=resolved_outputs_root,
            m6_root=resolved_m6_root,
        )
        if not stage_outputs:
            continue
        artifact_clip = _artifact_only_clip(scene_id, stage_outputs)
        if dataset_name != "all" and artifact_clip.dataset != dataset_name:
            continue
        merged.append(artifact_clip)

    return sorted(merged, key=lambda clip: (clip.dataset, clip.scene_id))


def clips_to_json(clips: Iterable[DatasetClip]) -> str:
    payload = []
    for clip in clips:
        record = asdict(clip)
        record["stage_outputs"] = clip.stage_output_paths
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
            "yes" if clip.trained else "no",
            clip.latest_stage or "-",
            _file_summary(clip),
            "; ".join(clip.notes) if clip.notes else "ok",
        )
        for clip in clips
    ]
    headers = (
        "dataset",
        "scene_id",
        "status",
        "trained",
        "latest_stage",
        "files",
        "notes",
    )
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
    if clip.dataset == "kitti360" and "left_images" in clip.files:
        return f"L={clip.files['left_images']} R={clip.files['right_images']}"
    if clip.dataset == "nvidia_ncore" and "components" in clip.files:
        return (
            f"components={clip.files['components']} "
            f"cameras={clip.files['camera_components']} "
            f"lidars={clip.files['lidar_components']}"
        )
    if clip.stage_output_paths:
        return f"artifacts={len(clip.stage_output_paths)}"
    return ""
