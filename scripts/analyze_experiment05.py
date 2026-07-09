#!/usr/bin/env python3

"""Analyze Experiment 05 seed-source comparison runs.

Reads the per-variant bundles produced by `scripts/experiment05-seed-comparison`
(`sgbm/<scene>` and `lidar/<scene>` run dirs) and writes a head-to-head
comparison of reconstruction quality between the SGBM-seeded and LiDAR-seeded
Octree-AnyGS models:

  - metrics_table.md / metrics.csv / comparison.json: PSNR/SSIM/LPIPS/GS_NUMS
    and seed point counts per variant, plus the lidar - sgbm delta per metric.
  - uncertainty_hist_overlay.png: per-anchor uncertainty distributions.
  - calibration_scatter_<variant>.png and sparsification_<metric>.png:
    per-view uncertainty against per-view rendering error.

Both variants share identical frames and cameras (enforced by the experiment
driver's fairness gate and re-checked here), so the full held-out test sets
are directly comparable.

Runs inside the vbogs-torch container (numpy + matplotlib); matplotlib is
imported lazily so the pure analysis functions work anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_experiment04 import (
    METRIC_TO_ERROR,
    METRICS,
    ause,
    build_test_name_mapping,
    find_run_dir,
    get_pyplot,
    locate_metrics_file,
    read_json,
    read_llffhold,
    select_method,
    sparsification_curve,
    spearman,
    write_metrics_csv,
)

DEFAULT_VARIANTS = ("sgbm", "lidar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Experiment root containing sgbm/<scene> and lidar/<scene> run dirs.",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help=(
            "Comma-separated variant dir names to analyze "
            f"(default: {','.join(DEFAULT_VARIANTS)})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory. Defaults to <experiment-root>/analysis.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plots (histogram, scatter, sparsification).",
    )
    return parser.parse_args()


def image_names_from_prepared_metadata(metadata: dict[str, Any]) -> list[str]:
    """Image names from either prepared-metadata shape.

    NCore frame records keep a ``cameras`` dict keyed by camera id; KITTI-360
    frame records keep an ``images`` list of per-camera entries.
    """
    names: list[str] = []
    for record in metadata.get("frame_records", []):
        cameras = record.get("cameras")
        if isinstance(cameras, dict):
            for camera_entry in cameras.values():
                names.append(str(camera_entry["image_name"]))
        else:
            for image_entry in record.get("images", []):
                names.append(str(image_entry["image_name"]))
    return names


@dataclass
class VariantData:
    name: str
    run_dir: Path
    seed_mode: str
    seed_points: int | None
    method: str
    results: dict[str, float]
    test_names: list[str]
    per_view: dict[str, dict[str, float]]  # metric -> stem -> value
    view_uncertainty: dict[str, float]  # stem -> score
    uncertainty_values: np.ndarray
    uncertainty_metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def load_variant(variant_dir: Path) -> VariantData:
    run_dir = find_run_dir(variant_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    prepared = read_json(run_dir / "prepared" / "metadata.json")
    llffhold = read_llffhold(run_dir / "octree" / "config.yaml")

    image_names = image_names_from_prepared_metadata(prepared)
    if not image_names:
        raise ValueError(f"No image names in prepared metadata for {variant_dir}")
    test_names = build_test_name_mapping(image_names, llffhold)

    results_all = read_json(locate_metrics_file(run_dir, manifest, "results.json"))
    per_view_all = read_json(locate_metrics_file(run_dir, manifest, "per_view.json"))
    method = select_method(results_all)
    warnings: list[str] = []

    per_view: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        positional = per_view_all[method].get(metric, {})
        if len(positional) != len(test_names):
            warnings.append(
                f"{variant_dir.name}: per_view {metric} has {len(positional)} entries, "
                f"expected {len(test_names)} test views"
            )
        by_stem: dict[str, float] = {}
        for key, value in positional.items():
            index = int(Path(key).name.split(".")[0])
            if index >= len(test_names):
                raise ValueError(
                    f"per_view key {key} out of range for {len(test_names)} test views"
                )
            by_stem[test_names[index]] = float(value)
        per_view[metric] = by_stem

    view_uncertainty: dict[str, float] = {}
    nbv_path = run_dir / "nbv" / "nbv_scores.json"
    if nbv_path.exists():
        nbv = read_json(nbv_path)
        for row in nbv.get("top_k", []):
            index = int(row["candidate_index"])
            if index < len(test_names) and row.get("image_name") != test_names[index]:
                raise ValueError(
                    f"NBV candidate {index} is {row.get('image_name')!r} but the "
                    f"reconstructed test list has {test_names[index]!r}; test-view "
                    "mapping is inconsistent"
                )
            view_uncertainty[str(row["image_name"])] = float(row["score"])
        if nbv.get("candidate_count", 0) > len(nbv.get("top_k", [])):
            warnings.append(
                f"{variant_dir.name}: nbv_scores.json is truncated "
                f"({len(nbv.get('top_k', []))}/{nbv.get('candidate_count')} views); "
                "rerun the nbv stage with a larger --nbv-top-k for full calibration"
            )
    else:
        warnings.append(f"{variant_dir.name}: no nbv/nbv_scores.json; skipping calibration")

    uncertainty_values = np.load(run_dir / "uncertainty" / "U.npy").astype(np.float64)
    uncertainty_metadata = read_json(run_dir / "uncertainty" / "uncertainty_metadata.json")

    seed_metadata = prepared.get("seed_metadata") or {}
    seed_points = seed_metadata.get("seed_point_count", prepared.get("stereo_max_points"))

    return VariantData(
        name=variant_dir.name,
        run_dir=run_dir,
        seed_mode=str(prepared.get("seed_mode", "unknown")),
        seed_points=int(seed_points) if seed_points is not None else None,
        method=method,
        results={key: float(value) for key, value in results_all[method].items()},
        test_names=test_names,
        per_view=per_view,
        view_uncertainty=view_uncertainty,
        uncertainty_values=uncertainty_values,
        uncertainty_metadata=uncertainty_metadata,
        warnings=warnings,
    )


def check_fairness(variants: list[VariantData]) -> list[str]:
    """All variants must share the identical full held-out test-view set."""
    reference: set[str] | None = None
    reference_variant = None
    for variant in variants:
        views = set(variant.test_names)
        if not views:
            raise ValueError(f"{variant.name}: no held-out test views")
        if reference is None:
            reference = views
            reference_variant = variant.name
        elif views != reference:
            missing = sorted(reference - views)
            extra = sorted(views - reference)
            raise ValueError(
                f"Test views differ between {reference_variant} and {variant.name} "
                f"(missing: {missing[:4]}..., extra: {extra[:4]}...); variants are "
                "not comparable. Re-run both variants with identical --frame-step/"
                "--max-frames (divisible by 8)."
            )
    return sorted(reference or set())


def variant_metric_rows(variants: list[VariantData]) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        row: dict[str, Any] = {
            "variant": variant.name,
            "seed_mode": variant.seed_mode,
            "seed_points": variant.seed_points,
            "method": variant.method,
            "test_views": len(variant.test_names),
            "GS_NUMS": variant.results.get("GS_NUMS", float("nan")),
        }
        for metric in METRICS:
            row[metric] = variant.results.get(metric, float("nan"))
        summary = variant.uncertainty_metadata
        anchor_count = int(summary.get("anchor_count", variant.uncertainty_values.shape[0]))
        observed = int(summary.get("observed_anchor_count", 0))
        row["anchors"] = anchor_count
        row["observed_anchor_fraction"] = observed / anchor_count if anchor_count else float("nan")
        row["U_mean"] = float(np.mean(variant.uncertainty_values))
        row["U_median"] = float(np.median(variant.uncertainty_values))
        observed_summary = summary.get("observed_summary", {})
        row["U_mean_observed"] = float(observed_summary.get("mean", float("nan")))
        rows.append(row)
    return rows


def seed_delta(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    """`lidar - sgbm` deltas for the headline metrics."""
    by_name = {row["variant"]: row for row in rows}
    if "sgbm" not in by_name or "lidar" not in by_name:
        return None
    delta: dict[str, float] = {}
    for metric in (*METRICS, "GS_NUMS"):
        sgbm_value = float(by_name["sgbm"].get(metric, float("nan")))
        lidar_value = float(by_name["lidar"].get(metric, float("nan")))
        delta[metric] = lidar_value - sgbm_value
    return delta


def calibration_summary(variant: VariantData) -> dict[str, Any] | None:
    if not variant.view_uncertainty:
        return None
    names = [
        name
        for name in variant.test_names
        if name in variant.view_uncertainty
        and all(name in variant.per_view[metric] for metric in METRICS)
    ]
    if len(names) < 2:
        return None
    uncertainty = np.array([variant.view_uncertainty[name] for name in names])
    summary: dict[str, Any] = {"views": len(names)}
    for metric in METRICS:
        errors = METRIC_TO_ERROR[metric](
            [variant.per_view[metric][name] for name in names]
        )
        fractions, predicted, oracle = sparsification_curve(uncertainty, errors)
        summary[metric] = {
            "spearman": spearman(uncertainty, errors),
            "ause": ause(predicted, oracle),
        }
        summary[f"_curve_{metric}"] = (fractions, predicted, oracle)
    summary["_names"] = names
    summary["_uncertainty"] = uncertainty
    return summary


def write_metrics_table(
    path: Path, rows: list[dict[str, Any]], delta: dict[str, float] | None
) -> None:
    columns = [
        ("variant", "{}"),
        ("seed_mode", "{}"),
        ("seed_points", "{}"),
        ("test_views", "{}"),
        ("PSNR", "{:.3f}"),
        ("SSIM", "{:.4f}"),
        ("LPIPS", "{:.4f}"),
        ("GS_NUMS", "{:.0f}"),
        ("observed_anchor_fraction", "{:.3f}"),
        ("U_mean", "{:.4g}"),
        ("U_mean_observed", "{:.4g}"),
    ]
    lines = [
        "# Experiment 05 metrics",
        "",
        "Both variants share the identical held-out test views; only the sparse",
        "seed source differs (sgbm = stereo SGBM depth, lidar = LiDAR scans).",
        "",
        "| " + " | ".join(name for name, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for name, fmt in columns:
            value = row.get(name, float("nan"))
            try:
                cells.append(fmt.format(value))
            except (ValueError, TypeError):
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    if delta is not None:
        lines.extend(
            [
                "",
                "## Delta (lidar - sgbm)",
                "",
                "| " + " | ".join(delta) + " |",
                "| " + " | ".join("---" for _ in delta) + " |",
                "| " + " | ".join(f"{value:+.4f}" for value in delta.values()) + " |",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_uncertainty_overlay(path: Path, variants: list[VariantData]) -> None:
    plt = get_pyplot()
    finite = np.concatenate(
        [
            variant.uncertainty_values[np.isfinite(variant.uncertainty_values)]
            for variant in variants
        ]
    )
    if finite.size == 0:
        return
    bins = np.linspace(float(finite.min()), float(finite.max()), 80)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant in variants:
        values = variant.uncertainty_values[np.isfinite(variant.uncertainty_values)]
        ax.hist(values, bins=bins, histtype="step", linewidth=1.5, label=variant.name)
    ax.set_yscale("log")
    ax.set_xlabel("per-anchor uncertainty U")
    ax.set_ylabel("anchor count")
    ax.set_title("Experiment 05: per-anchor uncertainty by seed source")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration_scatter(
    path: Path, variant: VariantData, summary: dict[str, Any]
) -> None:
    plt = get_pyplot()
    names = summary["_names"]
    uncertainty = summary["_uncertainty"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    for ax, metric in zip(np.atleast_1d(axes), METRICS):
        errors = METRIC_TO_ERROR[metric](
            [variant.per_view[metric][name] for name in names]
        )
        ax.scatter(uncertainty, errors, s=14, alpha=0.7)
        ax.set_xlabel("per-view uncertainty score")
        ax.set_ylabel(f"error ({metric})")
        ax.set_title(f"{metric}: rho={summary[metric]['spearman']:.3f}")
    fig.suptitle(f"{variant.name}: per-view uncertainty vs error")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sparsification(
    path: Path, metric: str, summaries: dict[str, dict[str, Any]]
) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant_name, summary in summaries.items():
        fractions, predicted, oracle = summary[f"_curve_{metric}"]
        (line,) = ax.plot(fractions, predicted, label=f"{variant_name} (predicted)")
        ax.plot(
            fractions,
            oracle,
            linestyle="--",
            alpha=0.6,
            color=line.get_color(),
            label=f"{variant_name} (oracle)",
        )
    ax.set_xlabel("fraction of most-uncertain views removed")
    ax.set_ylabel(f"mean error of retained views ({metric})")
    ax.set_title(f"Experiment 05 sparsification: {metric}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(
    experiment_root: Path,
    *,
    variants_filter: list[str] | None,
    output_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    wanted = variants_filter or list(DEFAULT_VARIANTS)
    variant_dirs = [
        experiment_root / name for name in wanted if (experiment_root / name).is_dir()
    ]
    if not variant_dirs:
        raise FileNotFoundError(
            f"No variant dirs ({', '.join(wanted)}) under {experiment_root}"
        )

    variants = [load_variant(path) for path in variant_dirs]
    shared_views = check_fairness(variants)
    rows = variant_metric_rows(variants)
    delta = seed_delta(rows)

    calibration: dict[str, dict[str, Any]] = {}
    for variant in variants:
        summary = calibration_summary(variant)
        if summary is not None:
            calibration[variant.name] = summary

    write_metrics_csv(output_dir / "metrics.csv", rows)
    write_metrics_table(output_dir / "metrics_table.md", rows, delta)

    if make_plots:
        plot_uncertainty_overlay(output_dir / "uncertainty_hist_overlay.png", variants)
        for variant in variants:
            if variant.name in calibration:
                plot_calibration_scatter(
                    output_dir / f"calibration_scatter_{variant.name}.png",
                    variant,
                    calibration[variant.name],
                )
        if calibration:
            for metric in METRICS:
                plot_sparsification(
                    output_dir / f"sparsification_{metric}.png", metric, calibration
                )

    warnings = [message for variant in variants for message in variant.warnings]
    comparison = {
        "experiment_root": str(experiment_root.resolve()),
        "fairness": {
            "shared_test_views": shared_views,
            "shared_test_view_count": len(shared_views),
        },
        "variants": rows,
        "delta_lidar_minus_sgbm": delta,
        "calibration": {
            name: {
                "views": summary["views"],
                **{metric: summary[metric] for metric in METRICS if metric in summary},
            }
            for name, summary in calibration.items()
        },
        "warnings": warnings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return comparison


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.experiment_root / "analysis")
    variants_filter = (
        [token.strip() for token in args.variants.split(",") if token.strip()]
        if args.variants
        else None
    )
    comparison = analyze(
        args.experiment_root,
        variants_filter=variants_filter,
        output_dir=output_dir,
        make_plots=not args.no_plots,
    )
    print(f"Analyzed {len(comparison['variants'])} variants -> {output_dir}")
    for row in comparison["variants"]:
        print(
            f"  {row['variant']}: seed_points={row['seed_points']} "
            f"PSNR={row['PSNR']:.3f} SSIM={row['SSIM']:.4f} LPIPS={row['LPIPS']:.4f} "
            f"U_mean={row['U_mean']:.4g}"
        )
    delta = comparison["delta_lidar_minus_sgbm"]
    if delta:
        print(
            "  delta (lidar - sgbm): "
            + " ".join(f"{metric}={value:+.4f}" for metric, value in delta.items())
        )
    for warning in comparison["warnings"]:
        print(f"  WARNING: {warning}")


if __name__ == "__main__":
    main()
