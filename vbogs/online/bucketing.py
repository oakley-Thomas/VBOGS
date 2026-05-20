"""Cached anchor-grid bucketing for online VBOGS updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class LevelLookup:
    level: int
    cur_size: float
    sorted_anchor_keys: np.ndarray
    sorted_anchor_ids: np.ndarray
    anchor_count: int


@dataclass(frozen=True)
class AnchorGridCache:
    """Search cache matching Octree-AnyGS anchor discretization."""

    anchor_xyz: np.ndarray
    anchor_level: np.ndarray
    voxel_size: float
    fork: int
    init_pos: np.ndarray
    levels: int
    lookups: tuple[LevelLookup, ...]

    @property
    def anchor_count(self) -> int:
        return int(self.anchor_xyz.shape[0])


def coords_to_keys(coords: np.ndarray) -> np.ndarray:
    coords = np.ascontiguousarray(coords, dtype=np.int64)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array")
    return coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).reshape(-1)


def _iter_chunk_ranges(total_count: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, total_count, chunk_size):
        yield start, min(start + chunk_size, total_count)


def build_anchor_grid_cache(
    *,
    anchor_xyz: np.ndarray,
    anchor_level: np.ndarray,
    voxel_size: float,
    fork: int,
    init_pos: np.ndarray,
    levels: int | None = None,
) -> AnchorGridCache:
    """Build per-level sorted lookup arrays for point-to-anchor assignment."""

    anchor_xyz = np.asarray(anchor_xyz, dtype=np.float32)
    anchor_level = np.asarray(anchor_level, dtype=np.int16).reshape(-1)
    init_pos = np.asarray(init_pos, dtype=np.float32).reshape(3)
    if anchor_xyz.ndim != 2 or anchor_xyz.shape[1] != 3:
        raise ValueError("anchor_xyz must have shape (N, 3)")
    if anchor_level.shape[0] != anchor_xyz.shape[0]:
        raise ValueError("anchor_level must have one row per anchor")
    if fork <= 0:
        raise ValueError("fork must be positive")
    if levels is None:
        levels = int(anchor_level.max()) + 1 if anchor_level.size else 0

    lookups: list[LevelLookup] = []
    for level in range(int(levels)):
        level_mask = anchor_level == level
        level_anchor_ids = np.nonzero(level_mask)[0].astype(np.int64)
        level_anchor_xyz = anchor_xyz[level_mask]
        cur_size = float(voxel_size) / (int(fork) ** level)
        if level_anchor_ids.size:
            grid = np.rint((level_anchor_xyz - init_pos[None, :]) / cur_size).astype(np.int64)
            keys = coords_to_keys(grid)
            order = np.argsort(keys)
            sorted_keys = keys[order]
            sorted_ids = level_anchor_ids[order]
        else:
            sorted_keys = coords_to_keys(np.empty((0, 3), dtype=np.int64))
            sorted_ids = np.empty((0,), dtype=np.int64)
        lookups.append(
            LevelLookup(
                level=level,
                cur_size=cur_size,
                sorted_anchor_keys=sorted_keys,
                sorted_anchor_ids=sorted_ids,
                anchor_count=int(level_anchor_ids.shape[0]),
            )
        )

    return AnchorGridCache(
        anchor_xyz=anchor_xyz,
        anchor_level=anchor_level,
        voxel_size=float(voxel_size),
        fork=int(fork),
        init_pos=init_pos,
        levels=int(levels),
        lookups=tuple(lookups),
    )


def _match_level_points(
    *,
    points_xyz_world: np.ndarray,
    init_pos: np.ndarray,
    cur_size: float,
    sorted_anchor_keys: np.ndarray,
    sorted_anchor_ids: np.ndarray,
    point_index_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sorted_anchor_ids.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    point_grid = np.rint((points_xyz_world - init_pos[None, :]) / cur_size).astype(np.int64)
    point_keys = coords_to_keys(point_grid)
    idx = np.searchsorted(sorted_anchor_keys, point_keys)
    within = idx < sorted_anchor_keys.shape[0]
    matched = np.zeros_like(within, dtype=bool)
    if np.any(within):
        matched_within = sorted_anchor_keys[idx[within]] == point_keys[within]
        matched[np.nonzero(within)[0]] = matched_within

    point_indices = (np.nonzero(matched)[0] + point_index_offset).astype(np.int64)
    matched_anchor_ids = sorted_anchor_ids[idx[matched]].astype(np.int64)
    return matched_anchor_ids, point_indices


def bucket_points_with_cache(
    points_xyz_world: np.ndarray,
    cache: AnchorGridCache,
    *,
    chunk_size: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Assign world-frame points to every matching Octree-AnyGS anchor level."""

    points_xyz_world = np.asarray(points_xyz_world, dtype=np.float32)
    if points_xyz_world.ndim != 2 or points_xyz_world.shape[1] != 3:
        raise ValueError("points_xyz_world must have shape (N, 3)")

    total_points = int(points_xyz_world.shape[0])
    matches_by_anchor: list[list[np.ndarray]] = [[] for _ in range(cache.anchor_count)]
    point_counts = np.zeros(cache.anchor_count, dtype=np.int64)
    level_assignment_counts = np.zeros(cache.levels, dtype=np.int64)

    for lookup in cache.lookups:
        for start, end in _iter_chunk_ranges(total_points, chunk_size):
            anchor_ids, point_indices = _match_level_points(
                points_xyz_world=points_xyz_world[start:end],
                init_pos=cache.init_pos,
                cur_size=lookup.cur_size,
                sorted_anchor_keys=lookup.sorted_anchor_keys,
                sorted_anchor_ids=lookup.sorted_anchor_ids,
                point_index_offset=start,
            )
            if anchor_ids.size == 0:
                continue
            counts = np.bincount(anchor_ids, minlength=cache.anchor_count).astype(np.int64)
            point_counts += counts
            level_assignment_counts[lookup.level] += int(anchor_ids.size)
            order = np.argsort(anchor_ids, kind="stable")
            sorted_anchor_ids = anchor_ids[order]
            sorted_point_indices = point_indices[order]
            unique_anchor_ids, run_starts, run_counts = np.unique(
                sorted_anchor_ids,
                return_index=True,
                return_counts=True,
            )
            for anchor_id, run_start, run_count in zip(
                unique_anchor_ids,
                run_starts,
                run_counts,
            ):
                matches_by_anchor[int(anchor_id)].append(
                    sorted_point_indices[run_start : run_start + run_count]
                )

    anchor_offsets = np.zeros(cache.anchor_count + 1, dtype=np.int64)
    anchor_offsets[1:] = np.cumsum(point_counts, dtype=np.int64)
    point_indices = np.empty(int(anchor_offsets[-1]), dtype=np.int64)
    cursor = 0
    for anchor_parts in matches_by_anchor:
        if anchor_parts:
            values = np.concatenate(anchor_parts).astype(np.int64, copy=False)
            point_indices[cursor : cursor + values.shape[0]] = values
            cursor += int(values.shape[0])

    if cursor != point_indices.shape[0]:
        raise RuntimeError("Online bucketing produced inconsistent packed offsets")

    touched_anchor_ids = np.nonzero(point_counts > 0)[0].astype(np.int64)
    return {
        "anchor_offsets": anchor_offsets,
        "point_indices": point_indices,
        "point_counts": point_counts.astype(np.int32),
        "touched_anchor_ids": touched_anchor_ids,
        "level_assignment_counts": level_assignment_counts,
    }


def save_anchor_grid_cache(path: Path, cache: AnchorGridCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "anchor_xyz": cache.anchor_xyz.astype(np.float32, copy=False),
        "anchor_level": cache.anchor_level.astype(np.int16, copy=False),
        "voxel_size": np.array(cache.voxel_size, dtype=np.float32),
        "fork": np.array(cache.fork, dtype=np.int16),
        "init_pos": cache.init_pos.astype(np.float32, copy=False),
        "levels": np.array(cache.levels, dtype=np.int16),
    }
    for lookup in cache.lookups:
        payload[f"level_{lookup.level}_cur_size"] = np.array(lookup.cur_size, dtype=np.float32)
        payload[f"level_{lookup.level}_sorted_anchor_keys"] = lookup.sorted_anchor_keys
        payload[f"level_{lookup.level}_sorted_anchor_ids"] = lookup.sorted_anchor_ids
        payload[f"level_{lookup.level}_anchor_count"] = np.array(lookup.anchor_count, dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_anchor_grid_cache(path: Path) -> AnchorGridCache:
    with np.load(path) as data:
        anchor_xyz = np.asarray(data["anchor_xyz"], dtype=np.float32)
        anchor_level = np.asarray(data["anchor_level"], dtype=np.int16)
        voxel_size = float(np.asarray(data["voxel_size"]))
        fork = int(np.asarray(data["fork"]))
        init_pos = np.asarray(data["init_pos"], dtype=np.float32)
        levels = int(np.asarray(data["levels"]))
        lookups = []
        for level in range(levels):
            lookups.append(
                LevelLookup(
                    level=level,
                    cur_size=float(np.asarray(data[f"level_{level}_cur_size"])),
                    sorted_anchor_keys=data[f"level_{level}_sorted_anchor_keys"],
                    sorted_anchor_ids=np.asarray(
                        data[f"level_{level}_sorted_anchor_ids"],
                        dtype=np.int64,
                    ),
                    anchor_count=int(np.asarray(data[f"level_{level}_anchor_count"])),
                )
            )

    return AnchorGridCache(
        anchor_xyz=anchor_xyz,
        anchor_level=anchor_level,
        voxel_size=voxel_size,
        fork=fork,
        init_pos=init_pos,
        levels=levels,
        lookups=tuple(lookups),
    )
