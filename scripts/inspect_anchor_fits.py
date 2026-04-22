#!/usr/bin/env python3

"""Inspect fitted anchor posteriors for the M4b human-validation step.

This script helps with the manual review called out in PLAN.md. It can:

- summarize the available M4 artifacts
- inspect one explicit anchor id
- suggest representative anchors by simple heuristics
- optionally export an anchor's assigned world-space points as a `.ply`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import unpack_group_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0008_sync",
        help="Drive id used to resolve default artifact paths.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Directory containing `points_norm.npz` and `pts_by_anchor.npz`.",
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=None,
        help="Posterior artifact to inspect. Defaults to full fit, then smoke fit.",
    )
    parser.add_argument(
        "--anchor-id",
        type=int,
        default=None,
        help="Inspect one explicit anchor id.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many anchors to show per heuristic list.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=5,
        help="How many assigned points to print in each sample table.",
    )
    parser.add_argument(
        "--export-ply",
        type=Path,
        default=None,
        help="Optional path for exporting the inspected anchor's assigned points as ASCII PLY.",
    )
    return parser.parse_args()


def resolve_bucket_root(args: argparse.Namespace) -> Path:
    if args.bucket_root is not None:
        return args.bucket_root.resolve()
    return (Path("data/m4") / args.drive).resolve()


def resolve_posterior_path(args: argparse.Namespace, bucket_root: Path) -> Path:
    if args.posterior is not None:
        return args.posterior.resolve()
    full_path = bucket_root / "anchor_posterior.npz"
    smoke_path = bucket_root / "anchor_posterior.smoke.npz"
    if full_path.exists():
        return full_path
    if smoke_path.exists():
        return smoke_path
    raise FileNotFoundError(
        f"Could not find `anchor_posterior.npz` or `anchor_posterior.smoke.npz` under {bucket_root}"
    )


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def build_observed_lookup(observed_anchor_ids: np.ndarray) -> Dict[int, int]:
    return {int(anchor_id): idx for idx, anchor_id in enumerate(observed_anchor_ids.tolist())}


def scalar_entropy_proxy(alpha_row: np.ndarray, spatial_kappa_row: np.ndarray, spatial_n_row: np.ndarray, final_k: int) -> float:
    """Cheap ranking proxy for 'uncertainty-like' anchors before M5 exists.

    This is not the Stage 4 entropy. It is only a rough heuristic for choosing
    anchors to inspect: lower kappa / lower n / flatter alpha tends to indicate
    broader or less-certain posteriors.
    """

    if final_k <= 0:
        return float("nan")
    alpha = alpha_row[:final_k]
    alpha = alpha / np.clip(alpha.sum(), 1e-8, np.inf)
    pi_flatness = -np.sum(alpha * np.log(np.clip(alpha, 1e-8, 1.0)))

    kappa = spatial_kappa_row[:final_k, 0, 0]
    n = spatial_n_row[:final_k, 0, 0]
    return float(pi_flatness + np.mean(1.0 / np.clip(kappa, 1e-8, np.inf)) + np.mean(1.0 / np.clip(n, 1e-8, np.inf)))


def format_vec(vec: np.ndarray, decimals: int = 4) -> str:
    arr = np.asarray(vec).reshape(-1)
    return "[" + ", ".join(f"{float(x):.{decimals}f}" for x in arr) + "]"


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb):
            handle.write(
                f"{float(point[0])} {float(point[1])} {float(point[2])} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def print_candidate_block(title: str, anchor_ids: np.ndarray, point_counts: np.ndarray, anchor_level: np.ndarray) -> None:
    print(title)
    for anchor_id in anchor_ids.tolist():
        print(
            f"  anchor={int(anchor_id):>7} level={int(anchor_level[anchor_id])} "
            f"pts={int(point_counts[anchor_id]):>6}"
        )


def inspect_anchor(
    anchor_id: int,
    *,
    pts_by_anchor: Dict[str, np.ndarray],
    points_norm_npz: Dict[str, np.ndarray],
    posterior_npz: Dict[str, np.ndarray],
    observed_lookup: Dict[int, int],
    export_ply: Path | None,
    sample_points: int,
) -> None:
    anchor_level = pts_by_anchor["anchor_level"]
    anchor_xyz = pts_by_anchor["anchor_xyz"]
    point_counts = pts_by_anchor["point_counts"]
    anchor_offsets = pts_by_anchor["anchor_offsets"]
    point_indices_all = pts_by_anchor["point_indices"]

    print("")
    print(f"Anchor {anchor_id}")
    print(f"  level        : {int(anchor_level[anchor_id])}")
    print(f"  anchor_xyz   : {format_vec(anchor_xyz[anchor_id], decimals=6)}")
    print(f"  point_count  : {int(point_counts[anchor_id])}")
    print(f"  observed     : {bool(posterior_npz['is_observed'][anchor_id])}")

    point_indices = unpack_group_slice(anchor_offsets, point_indices_all, anchor_id)
    points_world = points_norm_npz["points_world"][point_indices]
    points_norm = points_norm_npz["points_norm"][point_indices]
    frame_id = points_norm_npz["frame_id"][point_indices]
    rgb = points_norm_npz["rgb"][point_indices]

    sample_count = min(sample_points, point_indices.shape[0])
    if sample_count > 0:
        print("  sample assigned points:")
        for idx in range(sample_count):
            print(
                f"    idx={int(point_indices[idx]):>9} frame={int(frame_id[idx]):>6} "
                f"xyz_world={format_vec(points_world[idx, :3], decimals=4)} "
                f"rgb={format_vec(rgb[idx], decimals=1)}"
            )

    if anchor_id in observed_lookup:
        obs_idx = observed_lookup[anchor_id]
        final_k = int(posterior_npz["final_k"][obs_idx])
        final_elbo = float(posterior_npz["final_elbo"][obs_idx])
        selected_gain = float(posterior_npz["selected_gain"][obs_idx])
        under_modeled = bool(posterior_npz["under_modeled"][obs_idx])
        alpha = posterior_npz["alpha"][obs_idx, :final_k]
        alpha_norm = alpha / np.clip(alpha.sum(), 1e-8, np.inf)
        spatial_mean = posterior_npz["spatial_mean"][obs_idx, :final_k, :, 0]
        spatial_kappa = posterior_npz["spatial_kappa"][obs_idx, :final_k, 0, 0]
        spatial_n = posterior_npz["spatial_n"][obs_idx, :final_k, 0, 0]
        delta_mean = posterior_npz["delta_mean"][obs_idx, :final_k, :, 0]

        print("  posterior summary:")
        print(f"    final_k       : {final_k}")
        print(f"    final_elbo    : {final_elbo:.6g}")
        print(f"    selected_gain : {selected_gain:.6g}")
        print(f"    under_modeled : {under_modeled}")
        print(f"    alpha_norm    : {format_vec(alpha_norm, decimals=4)}")

        preview_components = min(3, final_k)
        print("    first components:")
        for comp in range(preview_components):
            print(
                f"      k={comp:>2} w={alpha_norm[comp]:.4f} "
                f"mu_xyz_norm={format_vec(spatial_mean[comp], decimals=4)} "
                f"kappa={float(spatial_kappa[comp]):.4f} "
                f"n={float(spatial_n[comp]):.4f} "
                f"mu_rgb_norm={format_vec(delta_mean[comp], decimals=4)}"
            )
    else:
        print("  posterior summary:")
        print("    anchor is unobserved under the current MIN_POINTS_PER_ANCHOR threshold")

    if export_ply is not None:
        write_ascii_ply(export_ply, points_world[:, :3], rgb)
        print(f"  exported points: {export_ply}")


def main() -> None:
    args = parse_args()
    bucket_root = resolve_bucket_root(args)
    posterior_path = resolve_posterior_path(args, bucket_root)

    pts_by_anchor = load_npz(bucket_root / "pts_by_anchor.npz")
    points_norm_npz = load_npz(bucket_root / "points_norm.npz")
    posterior_npz = load_npz(posterior_path)

    observed_anchor_ids = posterior_npz["observed_anchor_ids"].astype(np.int64)
    observed_lookup = build_observed_lookup(observed_anchor_ids)

    point_counts = pts_by_anchor["point_counts"]
    anchor_level = pts_by_anchor["anchor_level"]

    print(f"Bucket root : {bucket_root}")
    print(f"Posterior   : {posterior_path}")
    print(f"Anchors     : {anchor_level.shape[0]:,}")
    print(f"Observed    : {int(posterior_npz['is_observed'].sum()):,}")
    print(f"Fit rows     : {observed_anchor_ids.shape[0]:,}")

    if observed_anchor_ids.shape[0] > 0:
        proxy_scores = np.array(
            [
                scalar_entropy_proxy(
                    posterior_npz["alpha"][i],
                    posterior_npz["spatial_kappa"][i],
                    posterior_npz["spatial_n"][i],
                    int(posterior_npz["final_k"][i]),
                )
                for i in range(observed_anchor_ids.shape[0])
            ],
            dtype=np.float64,
        )

        richest_ids = np.argsort(point_counts)[-args.top_k :][::-1]
        sparsest_observed_ids = observed_anchor_ids[np.argsort(point_counts[observed_anchor_ids])[: args.top_k]]
        coarsest_observed_ids = observed_anchor_ids[np.argsort(anchor_level[observed_anchor_ids])[: args.top_k]]
        finest_observed_ids = observed_anchor_ids[np.argsort(anchor_level[observed_anchor_ids])[::-1][: args.top_k]]
        uncertain_order = np.argsort(proxy_scores)[::-1][: args.top_k]
        uncertain_ids = observed_anchor_ids[uncertain_order]

        print("")
        print_candidate_block("Top point-count anchors:", richest_ids, point_counts, anchor_level)
        print_candidate_block("Lowest-count observed anchors:", sparsest_observed_ids, point_counts, anchor_level)
        print_candidate_block("Coarsest observed anchors:", coarsest_observed_ids, point_counts, anchor_level)
        print_candidate_block("Finest observed anchors:", finest_observed_ids, point_counts, anchor_level)
        print_candidate_block("High uncertainty-proxy observed anchors:", uncertain_ids, point_counts, anchor_level)

    if args.anchor_id is not None:
        inspect_anchor(
            args.anchor_id,
            pts_by_anchor=pts_by_anchor,
            points_norm_npz=points_norm_npz,
            posterior_npz=posterior_npz,
            observed_lookup=observed_lookup,
            export_ply=args.export_ply.resolve() if args.export_ply is not None else None,
            sample_points=args.sample_points,
        )
    else:
        print("")
        print("No `--anchor-id` provided. Use one of the candidate ids above for a detailed inspection.")


if __name__ == "__main__":
    main()
