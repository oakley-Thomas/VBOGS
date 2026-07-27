"""Shared dynamic-object mask artifacts and geometry helpers.

The artifact uses ordinary greyscale PNGs: 255 is a static pixel that may be
used for training/geometry and 0 is a confirmed moving object.  Keeping this
format identical to Octree-AnyGS alpha masks makes it usable by both dataset
adapters without changes to the upstream submodule.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROAD_USER_COCO_IDS = frozenset((1, 2, 3, 4, 6, 7, 8))
COCO_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    6: "bus", 7: "train", 8: "truck",
}
ARTIFACT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_path(mask_root: Path, image_name: str | Path) -> Path:
    """Return the mask counterpart of a dataset-relative image name."""
    relative = Path(image_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Mask image name must be a safe relative path: {image_name}")
    return Path(mask_root) / "masks" / relative


def read_static_mask(mask_root: Path | None, image_name: str | Path, shape: Sequence[int]) -> np.ndarray:
    """Read a required static mask, or return all-static when masking is disabled."""
    height, width = int(shape[0]), int(shape[1])
    if mask_root is None:
        return np.ones((height, width), dtype=bool)
    path = mask_path(mask_root, image_name)
    if not path.is_file():
        raise FileNotFoundError(f"Dynamic mask missing for {image_name}: {path}")
    try:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except ImportError:  # pragma: no cover - OpenCV is present in torch runtime
        from PIL import Image
        image = np.asarray(Image.open(path).convert("L"))
    if image is None or image.shape != (height, width):
        found = None if image is None else image.shape
        raise ValueError(f"Dynamic mask {path} has shape {found}, expected {(height, width)}")
    return image > 0


def write_static_mask(mask_root: Path, image_name: str | Path, static_mask: np.ndarray) -> Path:
    path = mask_path(mask_root, image_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.where(np.asarray(static_mask, dtype=bool), 255, 0).astype(np.uint8)
    try:
        import cv2
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write mask {path}")
    except ImportError:  # pragma: no cover
        from PIL import Image
        Image.fromarray(image).save(path)
    return path


def write_overlay(mask_root: Path, image_name: str | Path, image_rgb: np.ndarray, dynamic_mask: np.ndarray) -> Path:
    """Save an inspectable red overlay without changing the source image."""
    relative = Path(image_name)
    path = Path(mask_root) / "overlays" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    dynamic = np.asarray(dynamic_mask, dtype=bool)
    if image.shape[:2] != dynamic.shape:
        raise ValueError("Overlay image and dynamic mask must have matching dimensions")
    image[dynamic] = (0.45 * image[dynamic] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    try:
        import cv2
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write overlay {path}")
    except ImportError:  # pragma: no cover
        from PIL import Image
        Image.fromarray(image).save(path)
    return path


def dilate_dynamic_mask(dynamic_mask: np.ndarray, pixels: int) -> np.ndarray:
    dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
    if pixels <= 0 or not dynamic_mask.any():
        return dynamic_mask.copy()
    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
        return cv2.dilate(dynamic_mask.astype(np.uint8), kernel).astype(bool)
    except ImportError:  # pragma: no cover
        padded = np.pad(dynamic_mask, pixels)
        result = np.zeros_like(dynamic_mask)
        for dy in range(2 * pixels + 1):
            for dx in range(2 * pixels + 1):
                result |= padded[dy : dy + dynamic_mask.shape[0], dx : dx + dynamic_mask.shape[1]]
        return result


@dataclass(frozen=True)
class InstanceObservation:
    frame_key: str
    timestamp_s: float
    class_id: int
    mask: np.ndarray
    centroid_world: np.ndarray | None
    score: float


@dataclass
class MotionTrack:
    track_id: int
    class_id: int
    observations: list[InstanceObservation]


def associate_world_tracks(
    observations: Iterable[InstanceObservation],
    *,
    gate_speed_mps: float = 40.0,
    base_gate_m: float = 1.0,
) -> list[MotionTrack]:
    """Associate reliable observations by class and ego-compensated world position."""
    tracks: list[MotionTrack] = []
    for observation in sorted(observations, key=lambda item: item.timestamp_s):
        if observation.centroid_world is None:
            continue
        candidate: MotionTrack | None = None
        best_distance = float("inf")
        for track in tracks:
            if track.class_id != observation.class_id or not track.observations:
                continue
            previous = track.observations[-1]
            assert previous.centroid_world is not None
            elapsed = max(0.0, observation.timestamp_s - previous.timestamp_s)
            gate = base_gate_m + gate_speed_mps * elapsed
            distance = float(np.linalg.norm(observation.centroid_world - previous.centroid_world))
            if distance <= gate and distance < best_distance:
                candidate, best_distance = track, distance
        if candidate is None:
            candidate = MotionTrack(track_id=len(tracks), class_id=observation.class_id, observations=[])
            tracks.append(candidate)
        candidate.observations.append(observation)
    return tracks


def track_motion_metrics(track: MotionTrack) -> dict[str, float | int]:
    reliable = [item for item in track.observations if item.centroid_world is not None]
    if len(reliable) < 2:
        return {"observation_count": len(reliable), "span_s": 0.0, "displacement_m": 0.0, "median_speed_mps": 0.0}
    positions = np.asarray([item.centroid_world for item in reliable], dtype=np.float64)
    times = np.asarray([item.timestamp_s for item in reliable], dtype=np.float64)
    # Use the first/last reliable world positions. The percentile form suppresses
    # outliers but underestimates short three-observation tracks by construction.
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    durations = np.diff(times)
    speeds = np.divide(steps, durations, out=np.zeros_like(steps), where=durations > 0)
    return {
        "observation_count": len(reliable),
        "span_s": float(times[-1] - times[0]),
        "displacement_m": displacement,
        "median_speed_mps": float(np.median(speeds)) if speeds.size else 0.0,
    }


def is_confirmed_moving(
    track: MotionTrack,
    *,
    min_observations: int = 3,
    min_span_s: float = 0.5,
    min_displacement_m: float = 1.0,
    min_speed_mps: float = 0.5,
) -> bool:
    values = track_motion_metrics(track)
    return bool(
        values["observation_count"] >= min_observations
        and values["span_s"] >= min_span_s
        and values["displacement_m"] >= min_displacement_m
        and values["median_speed_mps"] >= min_speed_mps
    )


def write_manifest(mask_root: Path, payload: dict[str, Any]) -> Path:
    mask_root = Path(mask_root)
    mask_root.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": ARTIFACT_VERSION, **payload}
    path = mask_root / "manifest.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(mask_root: Path) -> dict[str, Any]:
    path = Path(mask_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dynamic mask manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ARTIFACT_VERSION:
        raise ValueError(f"Unsupported dynamic mask manifest at {path}")
    return payload


def load_moving_cuboids(mask_root: Path, timestamp_us: int, tolerance_us: int = 100_000) -> list[dict[str, Any]]:
    """Get time-aligned NCore moving cuboids recorded by the mask stage."""
    rows = load_manifest(mask_root).get("moving_cuboids", [])
    grouped: dict[str, tuple[int, list[dict[str, Any]]]] = {}
    for row in rows:
        delta = abs(int(row.get("timestamp_us", -1)) - int(timestamp_us))
        if delta > tolerance_us:
            continue
        track_id = str(row.get("track_id", ""))
        previous = grouped.get(track_id)
        if previous is None or delta < previous[0]:
            grouped[track_id] = (delta, [row])
    return [values[1][0] for values in grouped.values()]


def points_in_cuboid(points_world: np.ndarray, cuboid: dict[str, Any]) -> np.ndarray:
    """Return a mask for a world-oriented cuboid serialized in the manifest."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(cuboid["center_world"], dtype=np.float64).reshape(3)
    size = np.asarray(cuboid["size_m"], dtype=np.float64).reshape(3)
    rotation = np.asarray(cuboid.get("rotation_world", np.eye(3)), dtype=np.float64).reshape(3, 3)
    local = (rotation.T @ (points - center).T).T
    return np.all(np.abs(local) <= (size * 0.5 + 1.0e-5), axis=1)


def filter_moving_cuboid_points(points_world: np.ndarray, mask_root: Path | None, timestamp_us: int) -> np.ndarray:
    points = np.asarray(points_world)
    if mask_root is None:
        return np.ones(points.shape[0], dtype=bool)
    keep = np.ones(points.shape[0], dtype=bool)
    for cuboid in load_moving_cuboids(mask_root, timestamp_us):
        keep &= ~points_in_cuboid(points, cuboid)
    return keep


def _tensor_to_numpy_mask(mask: Any) -> np.ndarray:
    value = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    return np.asarray(value, dtype=np.float32) >= 0.5


class TorchvisionMaskRCNNSegmenter:
    """Small injectable wrapper around a locally staged Mask R-CNN state dict."""

    def __init__(self, weights_path: Path, device: str = "cuda", score_threshold: float = 0.7) -> None:
        if not Path(weights_path).is_file():
            raise FileNotFoundError(
                f"Mask R-CNN weights are required and must be staged locally: {weights_path}"
            )
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.score_threshold = float(score_threshold)
        self.weights_sha256 = sha256_file(Path(weights_path))

    def detect(self, image_rgb: np.ndarray) -> list[tuple[int, float, np.ndarray]]:
        import torch
        image = torch.from_numpy(np.asarray(image_rgb, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
        with torch.inference_mode():
            output = self.model([image.to(self.device)])[0]
        result: list[tuple[int, float, np.ndarray]] = []
        for label, score, mask in zip(output["labels"], output["scores"], output["masks"]):
            class_id, confidence = int(label.item()), float(score.item())
            if class_id in ROAD_USER_COCO_IDS and confidence >= self.score_threshold:
                result.append((class_id, confidence, _tensor_to_numpy_mask(mask[0])))
        return result
