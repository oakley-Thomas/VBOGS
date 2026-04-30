#!/usr/bin/env python3

"""Convert saved M6 uncertainty/alpha arrays into PNG diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default="2013_05_28_drive_0008_sync")
    parser.add_argument(
        "--m6-root",
        type=Path,
        default=None,
        help="Directory containing `nbv_scores.json` and `top_images/`. Defaults to `data/m6/<drive>`.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing `rank_*_unc.npy` and `rank_*_alpha.npy`. Defaults to `<m6-root>/top_images`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG diagnostics are written. Defaults to `<m6-root>/viz`.",
    )
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=1.0,
        help="Lower percentile for uncertainty heatmap normalization.",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=99.0,
        help="Upper percentile for uncertainty heatmap normalization.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.01,
        help="Pixels below this alpha are treated as empty for uncertainty normalization.",
    )
    parser.add_argument(
        "--make-overlay",
        action="store_true",
        help="Also write uncertainty heatmaps with alpha encoded in the PNG alpha channel.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    m6_root = (args.m6_root or (Path("data/m6") / args.drive)).resolve()
    input_dir = (args.input_dir or (m6_root / "top_images")).resolve()
    output_dir = (args.output_dir or (m6_root / "viz")).resolve()
    return m6_root, input_dir, output_dir


def normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def turbo_like(x: np.ndarray) -> np.ndarray:
    """Compact blue-cyan-yellow-red colormap, good enough for diagnostics."""

    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(3.0 * x - 1.5), 0.0, 1.0)
    b = np.clip(1.2 - 1.8 * x, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    image = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGBA").save(path)


def rank_prefix(path: Path) -> str:
    name = path.name
    if not name.endswith("_unc.npy"):
        raise ValueError(f"Expected uncertainty filename ending in `_unc.npy`, got {name}")
    return name[: -len("_unc.npy")]


def main() -> None:
    args = parse_args()
    m6_root, input_dir, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_path = m6_root / "nbv_scores.json"
    scores = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {}

    rows = []
    for unc_path in sorted(input_dir.glob("rank_*_unc.npy")):
        prefix = rank_prefix(unc_path)
        alpha_path = input_dir / f"{prefix}_alpha.npy"
        if not alpha_path.exists():
            print(f"Skipping {unc_path}: missing {alpha_path.name}")
            continue

        unc = np.load(unc_path).astype(np.float32)
        alpha = np.load(alpha_path).astype(np.float32)
        if unc.shape != alpha.shape:
            raise ValueError(f"Shape mismatch for {prefix}: unc={unc.shape}, alpha={alpha.shape}")

        valid = np.isfinite(unc) & np.isfinite(alpha) & (alpha > args.alpha_threshold)
        if valid.any():
            lo = float(np.percentile(unc[valid], args.percentile_min))
            hi = float(np.percentile(unc[valid], args.percentile_max))
        else:
            lo = hi = float("nan")

        unc_norm = normalize(unc, lo, hi)
        alpha_norm = normalize(alpha, float(np.nanmin(alpha)), float(np.nanmax(alpha)))
        heat = turbo_like(unc_norm)
        alpha_gray = np.repeat(alpha_norm[..., None], 3, axis=-1)

        save_rgb(output_dir / f"{prefix}_unc_heat.png", heat)
        save_rgb(output_dir / f"{prefix}_alpha.png", alpha_gray)
        if args.make_overlay:
            save_rgba(output_dir / f"{prefix}_unc_alpha_overlay.png", heat, alpha_norm)

        rows.append(
            {
                "prefix": prefix,
                "unc_min": float(np.nanmin(unc)),
                "unc_max": float(np.nanmax(unc)),
                "unc_norm_min": lo,
                "unc_norm_max": hi,
                "alpha_min": float(np.nanmin(alpha)),
                "alpha_max": float(np.nanmax(alpha)),
                "valid_pixels": int(valid.sum()),
            }
        )

    summary = {
        "m6_root": str(m6_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "score_source": str(score_path) if score_path.exists() else None,
        "candidate_source": scores.get("candidate_source"),
        "candidate_count": scores.get("candidate_count"),
        "visualized_count": len(rows),
        "percentile_min": args.percentile_min,
        "percentile_max": args.percentile_max,
        "alpha_threshold": args.alpha_threshold,
        "items": rows,
    }
    (output_dir / "viz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} visualization set(s) to {output_dir}")


if __name__ == "__main__":
    main()
