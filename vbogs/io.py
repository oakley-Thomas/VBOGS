"""Repo-owned helpers for M4/M5 artifact I/O.

These utilities stay intentionally small so the PyTorch and JAX entry points
can share the same file conventions without importing each other's heavy
runtime stacks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def normalize_data_numpy(
    data: np.ndarray,
    data_params: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Numpy equivalent of `vbgs.data.utils.normalize_data`.

    M4a runs in the torch env, where we do not want to depend on the JAX stack
    just to standardize the 6D `(xyz, rgb)` features before writing them to
    disk for the JAX-side fit.
    """

    data = np.asarray(data, dtype=np.float32)
    if data_params is None:
        offset = data.mean(axis=0, keepdims=True)
        centered = data - offset
        stdevs = centered.std(axis=0, keepdims=True)
        stdevs = np.where(stdevs == 0, 1.0, stdevs)
        normalized = centered / stdevs
    else:
        offset = np.asarray(data_params["offset"], dtype=np.float32).reshape(1, -1)
        stdevs = np.asarray(data_params["stdevs"], dtype=np.float32).reshape(1, -1)
        stdevs = np.where(stdevs == 0, 1.0, stdevs)
        normalized = (data - offset) / stdevs

    params = {
        "offset": offset[0].astype(np.float32),
        "stdevs": stdevs[0].astype(np.float32),
    }
    return normalized.astype(np.float32), params


def pack_grouped_indices(
    group_ids: np.ndarray,
    values: np.ndarray,
    *,
    num_groups: int,
    value_dtype: np.dtype = np.int64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack sorted or unsorted `(group_id, value)` pairs into CSR-style arrays."""

    group_ids = np.asarray(group_ids, dtype=np.int64)
    values = np.asarray(values, dtype=value_dtype)

    if group_ids.shape != values.shape:
        raise ValueError("group_ids and values must have the same shape")
    if group_ids.ndim != 1:
        raise ValueError("group_ids and values must be 1D")
    if num_groups < 0:
        raise ValueError("num_groups must be non-negative")

    if group_ids.size == 0:
        offsets = np.zeros(num_groups + 1, dtype=np.int64)
        counts = np.zeros(num_groups, dtype=np.int64)
        return offsets, values.astype(value_dtype, copy=False), counts

    order = np.argsort(group_ids, kind="stable")
    sorted_group_ids = group_ids[order]
    sorted_values = values[order]

    counts = np.bincount(sorted_group_ids, minlength=num_groups).astype(np.int64)
    offsets = np.zeros(num_groups + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets, sorted_values, counts


def unpack_group_slice(
    offsets: np.ndarray,
    values: np.ndarray,
    group_id: int,
) -> np.ndarray:
    """Return the packed values belonging to a single group."""

    start = int(offsets[group_id])
    end = int(offsets[group_id + 1])
    return values[start:end]


def cap_group_counts(point_counts: np.ndarray, max_points_per_group: int) -> np.ndarray:
    """Return per-group counts after applying an optional fitting cap."""

    point_counts = np.asarray(point_counts, dtype=np.int32)
    if max_points_per_group <= 0:
        return point_counts.astype(np.int32, copy=True)
    return np.minimum(point_counts, max_points_per_group).astype(np.int32)


def select_group_values(
    offsets: np.ndarray,
    values: np.ndarray,
    group_id: int,
    *,
    max_values_per_group: int,
    seed: int,
) -> np.ndarray:
    """Return all or a deterministic random subset of packed group values."""

    group_values = unpack_group_slice(offsets, values, group_id)
    if max_values_per_group <= 0 or group_values.shape[0] <= max_values_per_group:
        return group_values

    rng = np.random.default_rng(seed + int(group_id))
    sampled = rng.choice(group_values, size=int(max_values_per_group), replace=False)
    sampled.sort()
    return sampled.astype(group_values.dtype, copy=False)
