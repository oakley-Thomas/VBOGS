#!/usr/bin/env python3

"""Consume online VBOGS batches and refresh touched-anchor uncertainty."""

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

from scripts.compute_uncertainty import compute_uncertainty
from vbogs.io import save_json
from vbogs.online import (
    apply_online_moment_update,
    atomic_save_npy,
    atomic_save_npz,
    load_npz_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Process pending batches once and exit.")
    parser.add_argument("--poll-sec", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--u-max", type=float, default=None)
    parser.add_argument("--min-points-per-anchor", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--update-mode",
        choices=("fixed-k-moment",),
        default="fixed-k-moment",
        help="Fixed-K online updater. Exact VBGS stat restoration can replace this contract later.",
    )
    return parser.parse_args()


def batch_seq(path: Path) -> int:
    return int(path.stem)


def pending_batches(batch_dir: Path, update_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(batch_dir.glob("*.npz"), key=batch_seq):
        update_path = update_dir / f"{path.stem}.npz"
        if not update_path.exists():
            paths.append(path)
    return paths


def process_batch(
    *,
    state_root: Path,
    batch_path: Path,
    eps: float,
    u_max: float | None,
    min_points_per_anchor: int | None,
) -> dict:
    start = time.perf_counter()
    state_path = state_root / "vbgs_online_state.npz"
    u_path = state_root / "U_online.npy"
    update_dir = state_root / "updates"
    state = load_npz_dict(state_path)
    batch = load_npz_dict(batch_path)
    seq = int(np.asarray(batch.get("seq", np.array(batch_seq(batch_path)))))
    min_points = (
        int(min_points_per_anchor)
        if min_points_per_anchor is not None
        else int(np.asarray(state.get("min_points_per_anchor", np.array(20))))
    )

    updated_state, update_metadata = apply_online_moment_update(
        state,
        batch,
        seq=seq,
        min_points_per_anchor=min_points,
        eps=eps,
    )
    uncertainty_result = compute_uncertainty(updated_state, u_max=u_max, eps=eps)
    uncertainty = np.asarray(uncertainty_result["uncertainty"], dtype=np.float32)

    atomic_save_npz(state_path, **updated_state)
    atomic_save_npy(u_path, uncertainty)

    elapsed = time.perf_counter() - start
    update_payload = {
        "seq": np.array(seq, dtype=np.int64),
        "updated_anchor_ids": np.asarray(update_metadata["updated_anchor_ids"], dtype=np.int64),
        "deferred_anchor_ids": np.asarray(update_metadata["deferred_anchor_ids"], dtype=np.int64),
        "updated_anchor_count": np.array(update_metadata["updated_anchor_count"], dtype=np.int32),
        "deferred_anchor_count": np.array(update_metadata["deferred_anchor_count"], dtype=np.int32),
        "elapsed_sec": np.array(elapsed, dtype=np.float32),
    }
    atomic_save_npz(update_dir / f"{seq:010d}.npz", **update_payload)
    metadata = {
        **update_metadata,
        "batch_path": str(batch_path),
        "state_path": str(state_path),
        "u_path": str(u_path),
        "elapsed_sec": elapsed,
        "uncertainty_summary": {
            "min": float(np.nanmin(uncertainty)),
            "max": float(np.nanmax(uncertainty)),
            "mean": float(np.nanmean(uncertainty)),
        },
    }
    save_json(update_dir / f"{seq:010d}.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    state_root = args.state_root.resolve()
    batch_dir = state_root / "batches"
    update_dir = state_root / "updates"
    update_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    while True:
        paths = pending_batches(batch_dir, update_dir)
        for path in paths:
            metadata = process_batch(
                state_root=state_root,
                batch_path=path,
                eps=args.eps,
                u_max=args.u_max,
                min_points_per_anchor=args.min_points_per_anchor,
            )
            processed += 1
            print(
                f"seq={metadata['seq']} updated={metadata['updated_anchor_count']} "
                f"deferred={metadata['deferred_anchor_count']} elapsed={metadata['elapsed_sec']:.4f}s"
            )
            if args.max_batches > 0 and processed >= args.max_batches:
                return
        if args.once:
            return
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
