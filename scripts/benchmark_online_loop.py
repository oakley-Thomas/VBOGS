#!/usr/bin/env python3

"""Replay online VBOGS update plumbing and report latency percentiles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.online_jax_updater import process_batch
from vbogs.io import save_json
from vbogs.online import (
    atomic_save_npz,
    bucket_points_with_cache,
    load_anchor_grid_cache,
    normalize_online_observations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--points-world", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--max-points-per-frame", type=int, default=150000)
    parser.add_argument("--point-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--run-updater", action="store_true")
    parser.add_argument(
        "--update-mode",
        choices=("fixed-k-moment", "exact-fixed-k"),
        default="fixed-k-moment",
    )
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def summarize(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def load_norm_params(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        "offset": np.asarray(raw["offset"], dtype=np.float32),
        "stdevs": np.asarray(raw["stdevs"], dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    state_root = args.state_root.resolve()
    cache = load_anchor_grid_cache(state_root / "anchor_grid_cache.npz")
    norm_params = load_norm_params(state_root / "norm_params.json")
    batch_dir = state_root / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.points_world) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = np.asarray(data["rgb"], dtype=np.uint8)
        frame_id = np.asarray(data["frame_id"], dtype=np.int32)

    frames = np.unique(frame_id)[: max(args.max_frames, 0)]
    rng = np.random.default_rng(0)
    rows = []
    for seq, frame in enumerate(frames.tolist()):
        frame_start = time.perf_counter()
        frame_mask = frame_id == frame
        frame_xyz = xyz[frame_mask]
        frame_rgb = rgb[frame_mask]
        stereo_ms = 0.0
        if args.max_points_per_frame > 0 and frame_xyz.shape[0] > args.max_points_per_frame:
            keep = rng.choice(frame_xyz.shape[0], size=args.max_points_per_frame, replace=False)
            frame_xyz = frame_xyz[keep]
            frame_rgb = frame_rgb[keep]

        t0 = time.perf_counter()
        points_norm, norm_stats = normalize_online_observations(
            frame_xyz,
            frame_rgb,
            norm_params,
        )
        normalization_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        bucketed = bucket_points_with_cache(
            frame_xyz,
            cache,
            chunk_size=args.point_chunk_size,
        )
        bucketing_ms = (time.perf_counter() - t0) * 1000.0

        batch_path = batch_dir / f"{seq:010d}.npz"
        atomic_save_npz(
            batch_path,
            seq=np.array(seq, dtype=np.int64),
            frame_id=np.array(frame, dtype=np.int32),
            points_norm=points_norm,
            anchor_offsets=bucketed["anchor_offsets"],
            point_indices=bucketed["point_indices"],
            point_counts=bucketed["point_counts"],
            touched_anchor_ids=bucketed["touched_anchor_ids"],
            normalization_outlier_count=np.array(norm_stats["outlier_count"], dtype=np.int32),
        )

        t0 = time.perf_counter()
        if args.run_updater:
            update_metadata = process_batch(
                state_root=state_root,
                batch_path=batch_path,
                eps=args.eps,
                u_max=None,
                min_points_per_anchor=None,
                update_mode=args.update_mode,
            )
            jax_update_ms = float(update_metadata["elapsed_sec"]) * 1000.0
        else:
            jax_update_ms = 0.0
        updater_wall_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - frame_start) * 1000.0
        rows.append(
            {
                "seq": seq,
                "frame_id": int(frame),
                "point_count": int(frame_xyz.shape[0]),
                "touched_anchor_count": int(bucketed["touched_anchor_ids"].shape[0]),
                "stereo_ms": stereo_ms,
                "normalization_ms": normalization_ms,
                "bucketing_ms": bucketing_ms,
                "jax_update_ms": jax_update_ms,
                "updater_wall_ms": updater_wall_ms,
                "gpu_refresh_ms": 0.0,
                "render_scoring_ms": 0.0,
                "total_ms": total_ms,
            }
        )
        print(
            f"seq={seq:04d} frame={frame} pts={frame_xyz.shape[0]:,} "
            f"touched={bucketed['touched_anchor_ids'].shape[0]:,} total={total_ms:.2f}ms"
        )

    summary = {
        "state_root": str(state_root),
        "points_world": str(args.points_world.resolve()),
        "update_mode": args.update_mode,
        "frame_count": len(rows),
        "acceptance_target_ms": 1000.0,
        "meets_1hz_target": summarize([row["total_ms"] for row in rows])["p95"] <= 1000.0,
        "latency_ms": {
            key: summarize([row[key] for row in rows])
            for key in (
                "stereo_ms",
                "normalization_ms",
                "bucketing_ms",
                "jax_update_ms",
                "updater_wall_ms",
                "gpu_refresh_ms",
                "render_scoring_ms",
                "total_ms",
            )
        },
        "frames": rows,
    }
    output = (args.output or (state_root / "online_benchmark.json")).resolve()
    save_json(output, summary)
    print(f"Wrote {output}")
    print("p95 total_ms={:.2f}".format(summary["latency_ms"]["total_ms"]["p95"]))


if __name__ == "__main__":
    main()
