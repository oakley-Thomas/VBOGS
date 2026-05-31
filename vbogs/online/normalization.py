"""Online normalization using fixed offline VBOGS normalization parameters."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def normalize_online_observations(
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    norm_params: Mapping[str, np.ndarray],
    *,
    outlier_z: float = 6.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Normalize online `(xyz, rgb)` observations with offline global params."""

    xyz_world = np.asarray(xyz_world, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if xyz_world.ndim != 2 or xyz_world.shape[1] != 3:
        raise ValueError("xyz_world must have shape (N, 3)")
    if rgb.shape != xyz_world.shape:
        raise ValueError("rgb must have shape (N, 3)")

    offset = np.asarray(norm_params["offset"], dtype=np.float32).reshape(1, 6)
    stdevs = np.asarray(norm_params["stdevs"], dtype=np.float32).reshape(1, 6)
    stdevs = np.where(stdevs == 0.0, 1.0, stdevs)
    raw = np.concatenate([xyz_world, rgb], axis=1).astype(np.float32)
    normalized = ((raw - offset) / stdevs).astype(np.float32)

    if normalized.size == 0:
        outlier_count = 0
        max_abs_z = 0.0
    else:
        abs_z = np.abs(normalized)
        outlier_count = int(np.count_nonzero(np.any(abs_z > outlier_z, axis=1)))
        max_abs_z = float(np.max(abs_z))

    stats = {
        "point_count": int(normalized.shape[0]),
        "outlier_count": outlier_count,
        "outlier_fraction": outlier_count / max(int(normalized.shape[0]), 1),
        "outlier_z": float(outlier_z),
        "max_abs_z": max_abs_z,
    }
    return normalized, stats
