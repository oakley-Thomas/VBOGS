"""Pose parsing helpers shared by VBOGS viewer entry points."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

POSE_CONVENTIONS = {"c2w", "w2c"}


def split_numeric_tokens(values: Any) -> list[float]:
    """Return numeric tokens from strings, flat sequences, or nested arrays."""

    tokens: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, np.ndarray):
            for item in value.reshape(-1).tolist():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if isinstance(value, str):
            for token in re.split(r"[\s,;]+", value.strip()):
                if token:
                    tokens.append(float(token))
            return
        tokens.append(float(value))

    visit(values)
    return tokens


def _validate_finite(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values")
    return matrix


def matrix_from_values(values: Any) -> np.ndarray:
    numbers = split_numeric_tokens(values)
    if len(numbers) == 16:
        matrix = np.asarray(numbers, dtype=np.float64).reshape(4, 4)
        return _validate_finite(matrix, "Pose matrix")
    if len(numbers) == 12:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :] = np.asarray(numbers, dtype=np.float64).reshape(3, 4)
        return _validate_finite(matrix, "Pose matrix")
    raise ValueError(f"Expected 16 values for a 4x4 matrix, or 12 values for a 3x4 matrix; got {len(numbers)}")


def coerce_pose_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return _validate_finite(matrix, "Pose matrix")
    if matrix.shape == (3, 4):
        padded = np.eye(4, dtype=np.float64)
        padded[:3, :] = matrix
        return _validate_finite(padded, "Pose matrix")
    if matrix.size in (12, 16):
        return matrix_from_values(matrix.reshape(-1).tolist())
    raise ValueError(f"Expected pose matrix shape (4, 4), (3, 4), 16, or 12 values; got {matrix.shape}")


def pose_to_c2w(matrix: np.ndarray, convention: str) -> np.ndarray:
    if convention not in POSE_CONVENTIONS:
        raise ValueError(f"Unsupported pose convention: {convention}")
    matrix = coerce_pose_matrix(matrix)
    c2w = matrix if convention == "c2w" else np.linalg.inv(matrix)
    return _validate_finite(np.asarray(c2w, dtype=np.float32), "Pose matrix")


def xyz_ypr_to_c2w(position: Any, yaw_pitch_roll_deg: Any) -> np.ndarray:
    xyz = split_numeric_tokens(position)
    ypr = split_numeric_tokens(yaw_pitch_roll_deg)
    if len(xyz) != 3:
        raise ValueError(f"Expected 3 position values, got {len(xyz)}")
    if len(ypr) != 3:
        raise ValueError(f"Expected 3 yaw/pitch/roll values, got {len(ypr)}")

    yaw, pitch, roll = (math.radians(float(value)) for value in ypr)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rz @ ry @ rx
    c2w[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return _validate_finite(c2w.astype(np.float32), "Pose matrix")


def values_to_c2w(values: Any, *, convention: str = "c2w") -> np.ndarray:
    numbers = split_numeric_tokens(values)
    if len(numbers) == 6:
        return xyz_ypr_to_c2w(numbers[:3], numbers[3:])
    if len(numbers) in (12, 16):
        return pose_to_c2w(matrix_from_values(numbers), convention)
    raise ValueError(
        "Expected 6 values for x y z yaw pitch roll, 12 values for a 3x4 matrix, "
        f"or 16 values for a 4x4 matrix; got {len(numbers)}"
    )


def dict_to_c2w(payload: dict[str, Any], *, fallback_convention: str = "c2w") -> np.ndarray:
    convention = str(payload.get("pose_convention", payload.get("convention", fallback_convention))).lower()
    if convention not in POSE_CONVENTIONS:
        raise ValueError(f"Unsupported pose convention: {convention}")

    if "c2w" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["c2w"]), "c2w")
    if "camera_to_world" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["camera_to_world"]), "c2w")
    if "w2c" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["w2c"]), "w2c")
    if "world_to_camera" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["world_to_camera"]), "w2c")
    if "world_view_transform" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["world_view_transform"]), "w2c")
    if "matrix" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["matrix"]), convention)
    if "pose" in payload:
        return parse_pose_to_c2w(payload["pose"], convention=convention)
    if "position" in payload and "yaw_pitch_roll_deg" in payload:
        return xyz_ypr_to_c2w(payload["position"], payload["yaw_pitch_roll_deg"])
    raise KeyError(
        "Pose payload must contain c2w, w2c, matrix, pose, or position with yaw_pitch_roll_deg"
    )


def parse_pose_to_c2w(value: Any, *, convention: str = "c2w") -> np.ndarray:
    if isinstance(value, dict):
        return dict_to_c2w(value, fallback_convention=convention)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            parsed = json.loads(stripped)
            return parse_pose_to_c2w(parsed, convention=convention)
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=object)
        if array.shape in ((4, 4), (3, 4)):
            return pose_to_c2w(coerce_pose_matrix(value), convention)
    return values_to_c2w(value, convention=convention)


def load_pose_file(path: Path, fallback_convention: str) -> tuple[np.ndarray, str]:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path)
        if np.asarray(value).size == 6:
            return parse_pose_to_c2w(value, convention=fallback_convention), "c2w"
        return coerce_pose_matrix(value), fallback_convention
    if suffix == ".npz":
        payload = np.load(path)
        for key in ("c2w", "camera_to_world"):
            if key in payload:
                return coerce_pose_matrix(payload[key]), "c2w"
        for key in ("w2c", "world_to_camera", "world_view_transform"):
            if key in payload:
                return coerce_pose_matrix(payload[key]), "w2c"
        if "position" in payload and "yaw_pitch_roll_deg" in payload:
            return xyz_ypr_to_c2w(payload["position"], payload["yaw_pitch_roll_deg"]), "c2w"
        if "pose" in payload:
            return parse_pose_to_c2w(payload["pose"], convention=fallback_convention), "c2w"
        if "matrix" in payload:
            return coerce_pose_matrix(payload["matrix"]), fallback_convention
        raise KeyError(f"{path} does not contain one of c2w, w2c, matrix, pose, or position/yaw_pitch_roll_deg")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return dict_to_c2w(payload, fallback_convention=fallback_convention), "c2w"
        return parse_pose_to_c2w(payload, convention=fallback_convention), "c2w"
    return parse_pose_to_c2w(path.read_text(encoding="utf-8").split(), convention=fallback_convention), "c2w"


def request_payload_to_c2w(payload: dict[str, Any], *, default_c2w: np.ndarray) -> np.ndarray:
    if "c2w" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["c2w"]), "c2w")
    if "w2c" in payload:
        return pose_to_c2w(coerce_pose_matrix(payload["w2c"]), "w2c")
    if "matrix" in payload:
        convention = str(payload.get("pose_convention", payload.get("convention", "c2w"))).lower()
        return pose_to_c2w(coerce_pose_matrix(payload["matrix"]), convention)
    if "pose" in payload:
        convention = str(payload.get("pose_convention", payload.get("convention", "c2w"))).lower()
        return parse_pose_to_c2w(payload["pose"], convention=convention)
    if "position" in payload and "yaw_pitch_roll_deg" in payload:
        return xyz_ypr_to_c2w(payload["position"], payload["yaw_pitch_roll_deg"])
    return pose_to_c2w(coerce_pose_matrix(default_c2w), "c2w")
