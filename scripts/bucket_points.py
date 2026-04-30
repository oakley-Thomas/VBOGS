#!/usr/bin/env python3

"""Bucket stereo point-cloud points into Octree-AnyGS anchors.

This is M4a from PLAN.md. It reads:

- a trained Octree-AnyGS run directory from M2
- a `points_world.npz` artifact from M3

and writes:

- `points_norm.npz` with normalized `(xyz, rgb)` rows for VBGS
- `pts_by_anchor.npz` with packed per-anchor point indices
- `norm_params.json` with the global normalization parameters
- `bucket_metadata.json` with diagnostics and provenance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import yaml
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import normalize_data_numpy, pack_grouped_indices, save_json

DEFAULT_OCTREE_OUTPUT_ROOT = Path("/data/OCTREE-ANYGS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0008_sync",
        help="Drive id used to resolve default input/output paths.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Octree-AnyGS run directory from M2. Defaults to the latest run under "
            "`/data/OCTREE-ANYGS/<drive>/`."
        ),
    )
    parser.add_argument(
        "--points-world",
        type=Path,
        default=None,
        help=(
            "M3 point cloud artifact. Defaults to "
            "`data/points_world/<drive>/points_world.npz`."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where the M4a artifacts will be written.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to load. `-1` selects the latest available iteration.",
    )
    parser.add_argument(
        "--octree-root",
        type=Path,
        default=Path("Octree-AnyGS"),
        help="Path to the Octree-AnyGS submodule.",
    )
    parser.add_argument(
        "--sample-anchor-count",
        type=int,
        default=5,
        help="How many non-empty anchors to print in the summary preview.",
    )
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path is not None:
        return args.model_path

    root = DEFAULT_OCTREE_OUTPUT_ROOT / args.drive
    if not root.exists():
        raise FileNotFoundError(f"No Octree-AnyGS output directory found at {root}")

    candidates = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "config.yaml").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No Octree-AnyGS runs found under {root}")
    return candidates[-1]


def resolve_points_world_path(args: argparse.Namespace) -> Path:
    if args.points_world is not None:
        return args.points_world
    return Path("data/points_world") / args.drive / "points_world.npz"


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    return Path("data/m4") / args.drive


def find_iteration_dir(model_path: Path, iteration: int) -> Tuple[int, Path]:
    point_cloud_root = model_path / "point_cloud"
    candidates = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iter_num = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        if (child / "point_cloud_anchor.ply").exists():
            candidates.append((iter_num, child))

    if not candidates:
        raise FileNotFoundError(f"No anchor checkpoints found under {point_cloud_root}")

    candidates.sort(key=lambda item: item[0])
    if iteration == -1:
        return candidates[-1]

    for iter_num, path in candidates:
        if iter_num == iteration:
            return iter_num, path
    raise FileNotFoundError(f"Iteration {iteration} not found under {point_cloud_root}")


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def reconstruct_grid_params(config: Dict, model_path: Path) -> Tuple[float, int, np.ndarray]:
    model_kwargs = config["model_params"]["model_config"]["kwargs"]
    base_layer = int(model_kwargs["base_layer"])
    source_path = Path(config["model_params"]["source_path"])
    input_ply = source_path / "sparse" / "0" / "points3D.ply"
    if not input_ply.exists():
        input_ply = model_path / "input.ply"
    if not input_ply.exists():
        raise FileNotFoundError(
            "Could not reconstruct Octree-AnyGS grid parameters because neither "
            f"{source_path / 'sparse/0/points3D.ply'} nor {model_path / 'input.ply'} exists."
        )

    ply = PlyData.read(str(input_ply))
    xyz = np.stack(
        [
            np.asarray(ply.elements[0]["x"], dtype=np.float32),
            np.asarray(ply.elements[0]["y"], dtype=np.float32),
            np.asarray(ply.elements[0]["z"], dtype=np.float32),
        ],
        axis=1,
    )

    extend = 1.1
    box_min = float(np.min(xyz)) * extend
    box_max = float(np.max(xyz)) * extend
    box_d = box_max - box_min
    voxel_size = float(box_d / (2 ** base_layer))
    init_pos = np.array([box_min, box_min, box_min], dtype=np.float32)
    fork = 2
    return voxel_size, fork, init_pos


def load_anchor_state(anchor_ply: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    ply = PlyData.read(str(anchor_ply))
    anchor_xyz = np.stack(
        [
            np.asarray(ply.elements[0]["x"], dtype=np.float32),
            np.asarray(ply.elements[0]["y"], dtype=np.float32),
            np.asarray(ply.elements[0]["z"], dtype=np.float32),
        ],
        axis=1,
    )
    anchor_level = np.asarray(ply.elements[0]["level"], dtype=np.int16)

    levels = None
    for info in ply.obj_info:
        if info.startswith("levels "):
            levels = int(round(float(info.split(" ", 1)[1])))
            break
    if levels is None:
        levels = int(anchor_level.max()) + 1
    return anchor_xyz, anchor_level, levels


def coords_to_keys(coords: np.ndarray) -> np.ndarray:
    coords = np.ascontiguousarray(coords, dtype=np.int64)
    return coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).reshape(-1)


def match_level_points(
    points_xyz_world: np.ndarray,
    init_pos: np.ndarray,
    cur_size: float,
    anchor_grid: np.ndarray,
    anchor_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    point_grid = np.rint((points_xyz_world - init_pos[None, :]) / cur_size).astype(np.int64)
    point_keys = coords_to_keys(point_grid)
    anchor_keys = coords_to_keys(anchor_grid)

    order = np.argsort(anchor_keys)
    sorted_anchor_keys = anchor_keys[order]
    sorted_anchor_ids = anchor_ids[order]

    idx = np.searchsorted(sorted_anchor_keys, point_keys)
    within = idx < sorted_anchor_keys.shape[0]
    matched = np.zeros_like(within, dtype=bool)
    if np.any(within):
        matched_within = sorted_anchor_keys[idx[within]] == point_keys[within]
        matched[np.nonzero(within)[0]] = matched_within

    point_indices = np.nonzero(matched)[0].astype(np.int64)
    matched_anchor_ids = sorted_anchor_ids[idx[matched]].astype(np.int64)
    return matched_anchor_ids, point_indices


def summarize_counts(point_counts: np.ndarray) -> Dict[str, float]:
    nonzero = point_counts[point_counts > 0]
    if nonzero.size == 0:
        return {
            "nonempty_anchor_count": 0,
            "count_min": 0,
            "count_p50": 0,
            "count_p90": 0,
            "count_p99": 0,
            "count_max": 0,
        }
    return {
        "nonempty_anchor_count": int(nonzero.size),
        "count_min": int(nonzero.min()),
        "count_p50": float(np.percentile(nonzero, 50)),
        "count_p90": float(np.percentile(nonzero, 90)),
        "count_p99": float(np.percentile(nonzero, 99)),
        "count_max": int(nonzero.max()),
    }


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args).resolve()
    points_world_path = resolve_points_world_path(args).resolve()
    output_root = resolve_output_root(args).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = load_yaml(model_path / "config.yaml")
    iteration, iteration_dir = find_iteration_dir(model_path, args.iteration)
    anchor_ply = iteration_dir / "point_cloud_anchor.ply"

    anchor_xyz, anchor_level, levels = load_anchor_state(anchor_ply)
    voxel_size, fork, init_pos = reconstruct_grid_params(config, model_path)

    world_npz = np.load(points_world_path)
    points_xyz_world = np.asarray(world_npz["xyz"], dtype=np.float32)
    points_rgb = np.asarray(world_npz["rgb"], dtype=np.uint8)
    frame_id = np.asarray(world_npz["frame_id"], dtype=np.int32)

    points_world = np.concatenate(
        [points_xyz_world, points_rgb.astype(np.float32)],
        axis=1,
    ).astype(np.float32)
    points_norm, norm_params = normalize_data_numpy(points_world)

    print(f"Loaded {points_xyz_world.shape[0]:,} world-frame points")
    print(f"Loaded {anchor_xyz.shape[0]:,} anchors across {levels} levels")

    matched_anchor_ids_all = []
    matched_point_indices_all = []

    for level in range(levels):
        level_mask = anchor_level == level
        level_anchor_ids = np.nonzero(level_mask)[0].astype(np.int64)
        level_anchor_xyz = anchor_xyz[level_mask]
        cur_size = voxel_size / (fork ** level)
        level_anchor_grid = np.rint((level_anchor_xyz - init_pos[None, :]) / cur_size).astype(np.int64)

        matched_anchor_ids, matched_point_indices = match_level_points(
            points_xyz_world=points_xyz_world,
            init_pos=init_pos,
            cur_size=cur_size,
            anchor_grid=level_anchor_grid,
            anchor_ids=level_anchor_ids,
        )
        matched_anchor_ids_all.append(matched_anchor_ids)
        matched_point_indices_all.append(matched_point_indices)
        print(
            f"Level {level:02d}: matched {matched_point_indices.shape[0]:,} "
            f"point-anchor assignments to {level_anchor_ids.shape[0]:,} anchors"
        )

    if matched_anchor_ids_all:
        group_ids = np.concatenate(matched_anchor_ids_all, axis=0)
        point_indices = np.concatenate(matched_point_indices_all, axis=0)
    else:
        group_ids = np.zeros((0,), dtype=np.int64)
        point_indices = np.zeros((0,), dtype=np.int64)

    anchor_offsets, packed_point_indices, point_counts = pack_grouped_indices(
        group_ids,
        point_indices,
        num_groups=anchor_xyz.shape[0],
        value_dtype=np.int64,
    )

    points_norm_path = output_root / "points_norm.npz"
    pts_by_anchor_path = output_root / "pts_by_anchor.npz"
    norm_params_path = output_root / "norm_params.json"
    metadata_path = output_root / "bucket_metadata.json"

    np.savez_compressed(
        points_norm_path,
        points_norm=points_norm,
        points_world=points_world,
        xyz_world=points_xyz_world,
        rgb=points_rgb,
        frame_id=frame_id,
    )
    np.savez_compressed(
        pts_by_anchor_path,
        anchor_offsets=anchor_offsets,
        point_indices=packed_point_indices,
        point_counts=point_counts.astype(np.int32),
        anchor_xyz=anchor_xyz,
        anchor_level=anchor_level,
        voxel_size=np.array(voxel_size, dtype=np.float32),
        fork=np.array(fork, dtype=np.int16),
        levels=np.array(levels, dtype=np.int16),
        init_pos=init_pos.astype(np.float32),
    )
    save_json(
        norm_params_path,
        {
            "offset": norm_params["offset"].tolist(),
            "stdevs": norm_params["stdevs"].tolist(),
        },
    )

    count_summary = summarize_counts(point_counts)
    nonempty_anchor_ids = np.nonzero(point_counts > 0)[0]
    preview_anchor_ids = nonempty_anchor_ids[: args.sample_anchor_count]
    preview = [
        {
            "anchor_id": int(anchor_id),
            "level": int(anchor_level[anchor_id]),
            "point_count": int(point_counts[anchor_id]),
            "anchor_xyz": anchor_xyz[anchor_id].round(6).tolist(),
        }
        for anchor_id in preview_anchor_ids
    ]

    save_json(
        metadata_path,
        {
            "drive": args.drive,
            "model_path": str(model_path),
            "iteration": iteration,
            "points_world_path": str(points_world_path),
            "anchor_count": int(anchor_xyz.shape[0]),
            "point_count": int(points_xyz_world.shape[0]),
            "assignment_count": int(packed_point_indices.shape[0]),
            "voxel_size": voxel_size,
            "fork": fork,
            "levels": levels,
            "init_pos": init_pos.tolist(),
            **count_summary,
            "preview_anchors": preview,
        },
    )

    print(f"Wrote {points_norm_path}")
    print(f"Wrote {pts_by_anchor_path}")
    print(f"Wrote {norm_params_path}")
    print(f"Wrote {metadata_path}")
    print("Per-anchor point-count summary:")
    print(
        "  non-empty={nonempty_anchor_count:,} min={count_min} p50={count_p50:.1f} "
        "p90={count_p90:.1f} p99={count_p99:.1f} max={count_max}".format(**count_summary)
    )
    for row in preview:
        print(
            f"  anchor {row['anchor_id']:>7} | level={row['level']} "
            f"| count={row['point_count']:>6} | xyz={row['anchor_xyz']}"
        )


if __name__ == "__main__":
    main()
