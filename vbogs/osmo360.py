"""Pure helpers for the RGB-only Osmo 360 reconstruction workflow.

The workflow deliberately consumes a *stitched equirectangular* video.  It
turns every sampled panorama into a six-camera virtual pinhole rig before
running COLMAP, which keeps Octree-AnyGS on its supported COLMAP input path.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


WORKFLOW = "osmo360_splat"
STAGES = ("validate", "project", "sfm", "prepare", "train", "render", "bundle")
PROFILE = "balanced"
MAX_UPLOAD_BYTES = 20 * 1024**3
MIN_DURATION_SECONDS = 30.0
MAX_DURATION_SECONDS = 12 * 60.0
CROP_SIZE = 1600
FOV_DEGREES = 100.0
FACE_SPECS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", -90.0, 0.0),
    ("up", 0.0, 90.0),
    ("down", 0.0, -90.0),
)
SCENE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class Osmo360ValidationError(ValueError):
    """Raised when an upload cannot enter the reconstruction workflow."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration_seconds: float
    codec_name: str
    video_stream_count: int


@dataclass(frozen=True)
class VirtualCamera:
    name: str
    yaw_degrees: float
    pitch_degrees: float
    width: int = CROP_SIZE
    height: int = CROP_SIZE
    fov_degrees: float = FOV_DEGREES

    @property
    def focal_length(self) -> float:
        return self.width / (2.0 * math.tan(math.radians(self.fov_degrees) / 2.0))

    def c2r(self) -> list[list[float]]:
        """Camera-from-rig rotation for COLMAP's JSON rig convention.

        The rig is centered at the panorama.  This matrix is explicit metadata
        for COLMAP; image creation uses the same yaw/pitch values.
        """
        yaw = math.radians(self.yaw_degrees)
        pitch = math.radians(self.pitch_degrees)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        return [
            [cy, -sy * sp, sy * cp],
            [0.0, cp, sp],
            [-sy, -cy * sp, cy * cp],
        ]


def virtual_cameras() -> tuple[VirtualCamera, ...]:
    return tuple(VirtualCamera(name, yaw, pitch) for name, yaw, pitch in FACE_SPECS)


def validate_scene_id(scene_id: str) -> str:
    value = str(scene_id).strip()
    if not SCENE_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise Osmo360ValidationError("Scene identifier must use letters, digits, dot, underscore, or dash (max 64 chars)")
    return value


def validate_upload_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise Osmo360ValidationError("Only stitched .mp4 and .mov videos are supported")
    return suffix


def parse_ffprobe(payload: dict[str, Any]) -> VideoInfo:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise Osmo360ValidationError("ffprobe output has no streams list")
    videos = [stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise Osmo360ValidationError("The upload must contain exactly one video stream")
    stream = videos[0]
    format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    try:
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(stream.get("duration") or format_info.get("duration"))
    except (KeyError, TypeError, ValueError) as exc:
        raise Osmo360ValidationError("ffprobe output is missing valid video dimensions or duration") from exc
    if width <= 0 or height <= 0 or not math.isfinite(duration):
        raise Osmo360ValidationError("Video dimensions and duration must be positive finite values")
    return VideoInfo(width, height, duration, str(stream.get("codec_name") or "unknown"), len(videos))


def validate_video_info(info: VideoInfo) -> None:
    if abs((info.width / info.height) - 2.0) > 0.02:
        raise Osmo360ValidationError("Video must be a stitched 2:1 equirectangular panorama")
    if not MIN_DURATION_SECONDS <= info.duration_seconds <= MAX_DURATION_SECONDS:
        raise Osmo360ValidationError("Video duration must be between 30 seconds and 12 minutes")


def sample_timestamps(duration_seconds: float) -> list[float]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    count = min(180, max(60, round(1.5 * duration_seconds)))
    # Avoid seeking exactly to the final frame, which commonly lands beyond a
    # variable-frame-rate stream's decodable range.
    return [round((index + 0.5) * duration_seconds / count, 6) for index in range(count)]


def split_timestamp_groups(frame_ids: Iterable[int]) -> dict[str, list[int]]:
    """Deterministically split whole panorama timestamps into 70/20/10 sets."""
    values = sorted({int(value) for value in frame_ids})
    total = len(values)
    counts = {"train": int(math.floor(total * 0.7)), "test": int(math.floor(total * 0.2))}
    counts["validation"] = total - counts["train"] - counts["test"]
    available = list(range(total))

    def uniform_take(count: int) -> list[int]:
        if count <= 0:
            return []
        chosen: list[int] = []
        for target in ((index + 0.5) * len(available) / count - 0.5 for index in range(count)):
            position = min((position for position in available if position not in chosen), key=lambda position: (abs(position - target), -position))
            chosen.append(position)
        return sorted(chosen)

    result: dict[str, list[int]] = {}
    for name in ("validation", "test"):
        positions = uniform_take(counts[name])
        result[name] = [values[position] for position in positions]
        selected = set(positions)
        available[:] = [position for position in available if position not in selected]
    result["train"] = [values[position] for position in available]
    return {name: result[name] for name in ("train", "test", "validation")}


def image_name(frame_index: int, camera: VirtualCamera) -> str:
    # COLMAP's rig configurator uses the identical basename in each sensor
    # directory to identify one simultaneous rig frame.
    return f"{camera.name}/frame_{frame_index:04d}.png"


def projection_command(*, ffmpeg: str, video: Path, timestamp: float, camera: VirtualCamera, output: Path) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        "v360=input=equirect:output=rectilinear"
        f":h_fov={camera.fov_degrees:g}:v_fov={camera.fov_degrees:g}"
        f":yaw={camera.yaw_degrees:g}:pitch={camera.pitch_degrees:g}"
        f",scale={camera.width}:{camera.height}"
    )
    return [ffmpeg, "-y", "-ss", f"{timestamp:.6f}", "-i", str(video), "-frames:v", "1", "-vf", vf, str(output)]


def rotation_to_quaternion(rotation: list[list[float]]) -> list[float]:
    """Return COLMAP's scalar-first quaternion for a 3×3 rotation matrix."""
    r = rotation
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        values = [0.25 * scale, (r[2][1] - r[1][2]) / scale, (r[0][2] - r[2][0]) / scale, (r[1][0] - r[0][1]) / scale]
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        scale = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        values = [(r[2][1] - r[1][2]) / scale, 0.25 * scale, (r[0][1] + r[1][0]) / scale, (r[0][2] + r[2][0]) / scale]
    elif r[1][1] > r[2][2]:
        scale = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        values = [(r[0][2] - r[2][0]) / scale, (r[0][1] + r[1][0]) / scale, 0.25 * scale, (r[1][2] + r[2][1]) / scale]
    else:
        scale = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        values = [(r[1][0] - r[0][1]) / scale, (r[0][2] + r[2][0]) / scale, (r[1][2] + r[2][1]) / scale, 0.25 * scale]
    return [float(value) for value in values]


def rig_config(_image_ids_by_camera: dict[str, Iterable[int]] | None = None) -> list[dict[str, Any]]:
    """Build COLMAP's documented folder-prefix rig configuration."""
    cameras = virtual_cameras()
    entries: list[dict[str, Any]] = []
    for index, camera in enumerate(cameras):
        entry: dict[str, Any] = {
            "image_prefix": f"{camera.name}/",
            "camera_model_name": "PINHOLE",
            "camera_params": [camera.focal_length, camera.focal_length, (camera.width - 1) / 2, (camera.height - 1) / 2],
        }
        if index == 0:
            entry["ref_sensor"] = True
        else:
            entry["cam_from_rig_rotation"] = rotation_to_quaternion(camera.c2r())
            entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        entries.append(entry)
    return [{"cameras": entries}]


def write_rig_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rig_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def profile_manifest(video: VideoInfo, timestamps: list[float]) -> dict[str, Any]:
    cameras = virtual_cameras()
    return {
        "workflow": WORKFLOW,
        "profile": PROFILE,
        "video": asdict(video),
        "projection": {
            "input": "equirectangular",
            "crop_size": CROP_SIZE,
            "fov_degrees": FOV_DEGREES,
            "timestamps": timestamps,
            "cameras": [asdict(camera) | {"focal_length": camera.focal_length} for camera in cameras],
        },
    }
