#!/usr/bin/env python3

"""Build an online VBOGS state bundle from validated offline artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json
from vbogs.online import (
    atomic_save_npy,
    atomic_save_npz,
    backfill_initial_fields_from_points,
    build_anchor_grid_cache,
    expand_posterior_to_anchor_rows,
    load_npz_dict,
    save_anchor_grid_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default="2013_05_28_drive_0008_sync")
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Offline M4/M5 artifact directory. Defaults to data/m4/<drive>.",
    )
    parser.add_argument("--posterior", type=Path, default=None)
    parser.add_argument("--u-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--update-mode",
        choices=("fixed-k-moment", "exact-fixed-k"),
        default="fixed-k-moment",
        help="Online posterior updater used by the JAX updater process.",
    )
    parser.add_argument(
        "--initial-mean-seed",
        type=int,
        default=0,
        help="Seed used when backfilling exact-fixed-k initial means for older M4b artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Online bundle root. Defaults to data/online/<drive>.",
    )
    parser.add_argument(
        "--validation-note",
        type=Path,
        default=None,
        help="Optional M7 validation note copied into the bundle manifest.",
    )
    parser.add_argument(
        "--allow-smoke-posterior",
        action="store_true",
        help="Allow using anchor_posterior.smoke.npz when a full posterior is absent.",
    )
    return parser.parse_args()


def resolve_bucket_root(args: argparse.Namespace) -> Path:
    if args.bucket_root is not None:
        return args.bucket_root.resolve()
    return (REPO_ROOT / "data" / "m4" / args.drive).resolve()


def resolve_posterior_path(bucket_root: Path, args: argparse.Namespace) -> Path:
    if args.posterior is not None:
        return args.posterior.resolve()
    full = bucket_root / "anchor_posterior.npz"
    if full.exists():
        return full
    smoke = bucket_root / "anchor_posterior.smoke.npz"
    if args.allow_smoke_posterior and smoke.exists():
        return smoke
    raise FileNotFoundError(
        f"Could not find full posterior at {full}. Pass --allow-smoke-posterior "
        f"to use {smoke} for plumbing tests."
    )


def main() -> None:
    args = parse_args()
    bucket_root = resolve_bucket_root(args)
    posterior_path = resolve_posterior_path(bucket_root, args)
    pts_by_anchor_path = bucket_root / "pts_by_anchor.npz"
    norm_params_path = bucket_root / "norm_params.json"
    u_path = (args.u_path or (bucket_root / "U.npy")).resolve()
    output_root = (args.output_root or (REPO_ROOT / "data" / "online" / args.drive)).resolve()

    if not pts_by_anchor_path.exists():
        raise FileNotFoundError(f"Missing M4 bucketing artifact: {pts_by_anchor_path}")
    if not norm_params_path.exists():
        raise FileNotFoundError(f"Missing normalization params: {norm_params_path}")
    if not u_path.exists():
        raise FileNotFoundError(f"Missing scalar uncertainty array: {u_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "batches").mkdir(exist_ok=True)
    (output_root / "updates").mkdir(exist_ok=True)

    pts_by_anchor = load_npz_dict(pts_by_anchor_path)
    posterior = load_npz_dict(posterior_path)
    u_values = np.asarray(np.load(u_path), dtype=np.float32).reshape(-1)
    anchor_count = int(pts_by_anchor["anchor_xyz"].shape[0])
    if u_values.shape[0] != anchor_count:
        raise ValueError(f"U length {u_values.shape[0]} does not match anchor count {anchor_count}")

    cache = build_anchor_grid_cache(
        anchor_xyz=pts_by_anchor["anchor_xyz"],
        anchor_level=pts_by_anchor["anchor_level"],
        voxel_size=float(np.asarray(pts_by_anchor["voxel_size"])),
        fork=int(np.asarray(pts_by_anchor["fork"])),
        init_pos=pts_by_anchor["init_pos"],
        levels=int(np.asarray(pts_by_anchor["levels"])),
    )
    save_anchor_grid_cache(output_root / "anchor_grid_cache.npz", cache)

    online_state = expand_posterior_to_anchor_rows(posterior)
    backfilled_initial_state = False
    if args.update_mode == "exact-fixed-k":
        initial_present = np.isfinite(online_state["initial_spatial_mean"]).any(axis=(1, 2, 3))
        missing_initial = bool(np.any(np.asarray(online_state["fit_completed"], dtype=bool) & ~initial_present))
        if missing_initial:
            points_norm_path = bucket_root / "points_norm.npz"
            if not points_norm_path.exists():
                raise FileNotFoundError(
                    f"Missing normalized points needed to backfill exact VBGS state: {points_norm_path}"
                )
            points_norm = np.asarray(load_npz_dict(points_norm_path)["points_norm"], dtype=np.float32)
            online_state = backfill_initial_fields_from_points(
                online_state,
                points_norm=points_norm,
                anchor_offsets=np.asarray(pts_by_anchor["anchor_offsets"], dtype=np.int64),
                point_indices=np.asarray(pts_by_anchor["point_indices"], dtype=np.int64),
                seed=args.initial_mean_seed,
            )
            backfilled_initial_state = True
    atomic_save_npz(output_root / "vbgs_online_state.npz", **online_state)
    atomic_save_npy(output_root / "U_online.npy", u_values)
    shutil.copy2(norm_params_path, output_root / "norm_params.json")

    validation_note_text = None
    if args.validation_note is not None:
        validation_note_text = args.validation_note.read_text(encoding="utf-8")

    manifest = {
        "drive": args.drive,
        "state_version": int(online_state["state_version"]),
        "model_path": str(args.model_path.resolve()) if args.model_path is not None else None,
        "source_bucket_root": str(bucket_root),
        "source_posterior": str(posterior_path),
        "source_u_path": str(u_path),
        "anchor_count": anchor_count,
        "initial_observed_anchor_count": int(np.asarray(posterior["is_observed"], dtype=bool).sum()),
        "online_state": "vbgs_online_state.npz",
        "uncertainty": "U_online.npy",
        "anchor_grid_cache": "anchor_grid_cache.npz",
        "norm_params": "norm_params.json",
        "batch_dir": "batches",
        "update_dir": "updates",
        "update_mode": args.update_mode.replace("-", "_"),
        "fixed_k": True,
        "online_scene_retraining": False,
        "empty_space_prior": False,
        "backfilled_initial_state": backfilled_initial_state,
        "validation_note": validation_note_text,
    }
    save_json(output_root / "online_manifest.json", manifest)

    print(f"Wrote {output_root / 'online_manifest.json'}")
    print(f"Wrote {output_root / 'anchor_grid_cache.npz'}")
    print(f"Wrote {output_root / 'vbgs_online_state.npz'}")
    print(f"Wrote {output_root / 'U_online.npy'}")


if __name__ == "__main__":
    main()
