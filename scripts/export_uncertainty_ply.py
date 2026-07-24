#!/usr/bin/env python3

"""Write a viewer-friendly anchor point cloud colored by per-anchor uncertainty."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-ply", type=Path, required=True)
    parser.add_argument("--u-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help=(
            "Upper percentile of finite uncertainty used as the color ceiling. "
            "Unobserved anchors sit at the maximum and would otherwise flatten the ramp."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.clip_percentile <= 100.0:
        raise ValueError("--clip-percentile must be in (0, 100]")

    plydata = PlyData.read(args.anchor_ply)
    vertex = plydata.elements[0]
    xyz = np.stack(
        (
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ),
        axis=1,
    ).astype(np.float32)

    uncertainty = np.load(args.u_path).astype(np.float32).reshape(-1)
    if uncertainty.shape[0] != xyz.shape[0]:
        raise ValueError(
            "Uncertainty length does not match the anchor count "
            f"({uncertainty.shape[0]} vs {xyz.shape[0]}); "
            "U.npy and the anchor ply must come from the same selection lock"
        )

    finite = np.isfinite(uncertainty)
    if not finite.any():
        raise ValueError(f"No finite uncertainty values in {args.u_path}")
    low = float(uncertainty[finite].min())
    high = float(np.percentile(uncertainty[finite], args.clip_percentile))
    if not high > low:
        high = float(uncertainty[finite].max())
    span = high - low if high > low else 1.0

    normalized = np.clip((np.nan_to_num(uncertainty, nan=high, posinf=high, neginf=low) - low) / span, 0.0, 1.0)

    from matplotlib import colormaps

    colors = colormaps[args.colormap](normalized)[:, :3]
    rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("uncertainty", "f4"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"], elements["y"], elements["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements["red"], elements["green"], elements["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    elements["uncertainty"] = uncertainty

    args.output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(elements, "vertex")],
        obj_info=[
            f"uncertainty_color_low {low:.8g}",
            f"uncertainty_color_high {high:.8g}",
            f"uncertainty_colormap {args.colormap}",
        ],
    ).write(args.output)
    print(
        f"Wrote {args.output} ({xyz.shape[0]} anchors, color range [{low:.6g}, {high:.6g}])"
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
