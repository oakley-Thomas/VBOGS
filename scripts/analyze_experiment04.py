#!/usr/bin/env python3

"""Analyze Experiment 04 camera-count comparison runs.

Reads the per-variant bundles produced by `scripts/experiment04-camera-count`
(one `cam<N>/<scene>` run dir per camera-count variant) and writes a
comparison of reconstruction quality and VB uncertainty:

  - metrics_table.md / metrics.csv / comparison.json: PSNR/SSIM/LPIPS/GS_NUMS
    per variant, overall and restricted to the shared primary-camera test
    views (the like-for-like comparison across camera counts).
  - uncertainty_hist_overlay.png: per-anchor uncertainty distributions.
  - calibration_scatter_<variant>.png and sparsification_<metric>.png:
    per-view uncertainty (alpha-normalized render of per-anchor U from
    `nbv_scores.json`) against per-view rendering error, with Spearman
    correlations and sparsification (AUSE) summaries.

Runs inside the vbogs-torch container (numpy + matplotlib); matplotlib is
imported lazily so the pure analysis functions work anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PRIMARY_CAMERA = "camera_front_wide_120fov"
METRICS = ("PSNR", "SSIM", "LPIPS")
# Convert each quality metric into an "error" (higher is worse) for
# calibration/sparsification analysis.
METRIC_TO_ERROR = {
    "PSNR": lambda values: -np.asarray(values, dtype=np.float64),
    "SSIM": lambda values: 1.0 - np.asarray(values, dtype=np.float64),
    "LPIPS": lambda values: np.asarray(values, dtype=np.float64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Experiment root containing cam<N>/<scene> variant run dirs.",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variant dir names to analyze (default: every cam* dir).",
    )
    parser.add_argument(
        "--primary-camera",
        default=DEFAULT_PRIMARY_CAMERA,
        help="Camera whose held-out test views are shared across variants.",
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_llffhold(config_path: Path) -> int:
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return int(config["model_params"]["llffhold"])


def image_names_from_prepared_metadata(metadata: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for record in metadata.get("frame_records", []):
        for camera_entry in record.get("cameras", {}).values():
            names.append(str(camera_entry["image_name"]))
    return names


def build_test_name_mapping(image_names: list[str], llffhold: int) -> list[str]:
    """Test-view stems in Octree-AnyGS order.

    Octree-AnyGS sorts cameras by image path and holds out every
    ``llffhold``-th index; positional per_view.json key ``{k:05d}.png``
    corresponds to entry ``k`` of the returned list. Stems match
    ``Camera.image_name`` (basename without extension).
    """
    ordered = sorted(image_names)
    test_names = ordered[0::llffhold]
    return [Path(name).name.split(".")[0] for name in test_names]


def wide_only_names(test_names: list[str], primary_camera: str) -> list[str]:
    prefix = f"{primary_camera}_"
    return [name for name in test_names if name.startswith(prefix)]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties (no scipy)."""

    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.shape[0], dtype=np.float64)
        sorted_values = values[order]
        index = 0
        while index < sorted_values.shape[0]:
            tie_end = index
            while (
                tie_end + 1 < sorted_values.shape[0]
                and sorted_values[tie_end + 1] == sorted_values[index]
            ):
                tie_end += 1
            ranks[order[index : tie_end + 1]] = 0.5 * (index + tie_end) + 1.0
            index = tie_end + 1
        return ranks

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2:
        return float("nan")
    rank_x = average_ranks(x)
    rank_y = average_ranks(y)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denom = np.sqrt((rank_x**2).sum() * (rank_y**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((rank_x * rank_y).sum() / denom)


def sparsification_curve(
    uncertainty: np.ndarray, errors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean error of retained views as the most-uncertain views are removed.

    Returns ``(fractions_removed, predicted_curve, oracle_curve)``. The
    predicted curve removes views by descending predicted uncertainty, the
    oracle by descending actual error; a well-calibrated predictor tracks
    the oracle.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    n = errors.shape[0]
    if n == 0:
        raise ValueError("sparsification_curve needs at least one view")

    def curve(order: np.ndarray) -> np.ndarray:
        # order sorts views from most to least removable; entry k of the
        # curve keeps the n-k least-removable views.
        kept_errors = errors[order[::-1]]
        cumulative = np.cumsum(kept_errors)
        kept_counts = np.arange(1, n + 1, dtype=np.float64)
        means = cumulative / kept_counts
        return means[::-1]

    predicted = curve(np.argsort(-uncertainty, kind="mergesort"))
    oracle = curve(np.argsort(-errors, kind="mergesort"))
    fractions = np.arange(n, dtype=np.float64) / n
    return fractions, predicted, oracle


def ause(predicted: np.ndarray, oracle: np.ndarray) -> float:
    """Area between the predicted and oracle sparsification curves."""
    return float(np.mean(np.asarray(predicted) - np.asarray(oracle)))


@dataclass
class VariantData:
    name: str
    run_dir: Path
    camera_ids: list[str]
    method: str
    results: dict[str, float]
    test_names: list[str]
    per_view: dict[str, dict[str, float]]  # metric -> stem -> value
    view_uncertainty: dict[str, float]  # stem -> score
    uncertainty_values: np.ndarray
    uncertainty_metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def find_run_dir(variant_dir: Path) -> Path:
    if (variant_dir / "run_manifest.json").exists():
        return variant_dir
    candidates = sorted(
        path
        for path in variant_dir.iterdir()
        if path.is_dir() and (path / "run_manifest.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No run_manifest.json under {variant_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple run dirs under {variant_dir}: {candidates}")
    return candidates[0]


def locate_metrics_file(run_dir: Path, manifest: dict[str, Any], name: str) -> Path:
    bundled = run_dir / "octree" / name
    if bundled.exists():
        return bundled
    model_path = Path(manifest["source_paths"]["octree_model_path"])
    fallback = model_path / name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"{name} not found in bundle ({bundled}) or model path ({fallback})"
    )


def select_method(results: dict[str, Any]) -> str:
    methods = sorted(results)
    if not methods:
        raise ValueError("results.json contains no methods")

    def iteration_key(method: str) -> int:
        tail = method.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    return max(methods, key=iteration_key)


def load_variant(variant_dir: Path, primary_camera: str) -> VariantData:
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

    return VariantData(
        name=variant_dir.name,
        run_dir=run_dir,
        camera_ids=[str(camera_id) for camera_id in prepared.get("camera_ids", [])],
        method=method,
        results={key: float(value) for key, value in results_all[method].items()},
        test_names=test_names,
        per_view=per_view,
        view_uncertainty=view_uncertainty,
        uncertainty_values=uncertainty_values,
        uncertainty_metadata=uncertainty_metadata,
        warnings=warnings,
    )


def check_fairness(variants: list[VariantData], primary_camera: str) -> list[str]:
    """All variants must share identical primary-camera test views."""
    reference = None
    reference_variant = None
    for variant in variants:
        wide = set(wide_only_names(variant.test_names, primary_camera))
        if not wide:
            raise ValueError(
                f"{variant.name}: no test views from primary camera {primary_camera!r}"
            )
        if reference is None:
            reference = wide
            reference_variant = variant.name
        elif wide != reference:
            missing = sorted(reference - wide)
            extra = sorted(wide - reference)
            raise ValueError(
                f"Primary-camera test views differ between {reference_variant} and "
                f"{variant.name} (missing: {missing[:4]}..., extra: {extra[:4]}...); "
                "variants are not comparable. Re-run with identical --frame-step/"
                "--max-frames (divisible by 8) for every variant."
            )
    return sorted(reference or set())


def mean_over(per_view: dict[str, float], names: list[str]) -> float:
    values = [per_view[name] for name in names if name in per_view]
    if not values:
        return float("nan")
    return float(np.mean(values))


def variant_metric_rows(
    variants: list[VariantData], primary_camera: str
) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        wide = wide_only_names(variant.test_names, primary_camera)
        row: dict[str, Any] = {
            "variant": variant.name,
            "cameras": len(variant.camera_ids),
            "camera_ids": ",".join(variant.camera_ids),
            "method": variant.method,
            "test_views": len(variant.test_names),
            "wide_test_views": len(wide),
            "GS_NUMS": variant.results.get("GS_NUMS", float("nan")),
        }
        for metric in METRICS:
            row[metric] = variant.results.get(metric, float("nan"))
            row[f"{metric}_wide"] = mean_over(variant.per_view[metric], wide)
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


def calibration_summary(
    variant: VariantData, primary_camera: str
) -> dict[str, Any] | None:
    if not variant.view_uncertainty:
        return None
    wide = set(wide_only_names(variant.test_names, primary_camera))
    summary: dict[str, Any] = {"views": 0}
    names = [
        name
        for name in variant.test_names
        if name in variant.view_uncertainty
        and all(name in variant.per_view[metric] for metric in METRICS)
    ]
    if len(names) < 2:
        return None
    uncertainty = np.array([variant.view_uncertainty[name] for name in names])
    summary["views"] = len(names)
    wide_indices = np.array([name in wide for name in names], dtype=bool)
    for metric in METRICS:
        errors = METRIC_TO_ERROR[metric](
            [variant.per_view[metric][name] for name in names]
        )
        fractions, predicted, oracle = sparsification_curve(uncertainty, errors)
        summary[metric] = {
            "spearman": spearman(uncertainty, errors),
            "spearman_wide": (
                spearman(uncertainty[wide_indices], errors[wide_indices])
                if int(wide_indices.sum()) >= 2
                else float("nan")
            ),
            "ause": ause(predicted, oracle),
        }
        summary[f"_curve_{metric}"] = (fractions, predicted, oracle)
    summary["_names"] = names
    summary["_uncertainty"] = uncertainty
    return summary


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        ("variant", "{}"),
        ("cameras", "{}"),
        ("test_views", "{}"),
        ("PSNR", "{:.3f}"),
        ("PSNR_wide", "{:.3f}"),
        ("SSIM", "{:.4f}"),
        ("SSIM_wide", "{:.4f}"),
        ("LPIPS", "{:.4f}"),
        ("LPIPS_wide", "{:.4f}"),
        ("GS_NUMS", "{:.0f}"),
        ("observed_anchor_fraction", "{:.3f}"),
        ("U_mean", "{:.4g}"),
        ("U_mean_observed", "{:.4g}"),
    ]
    lines = [
        "# Experiment 04 metrics",
        "",
        "`*_wide` columns are restricted to the shared primary-camera test",
        "views and are the like-for-like comparison across camera counts.",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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
    ax.set_title("Experiment 04: per-anchor uncertainty by camera count")
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
    ax.set_title(f"Experiment 04 sparsification: {metric}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(
    experiment_root: Path,
    *,
    variants_filter: list[str] | None,
    primary_camera: str,
    output_dir: Path,
    make_plots: bool,
) -> dict[str, Any]:
    variant_dirs = sorted(
        path
        for path in experiment_root.iterdir()
        if path.is_dir() and path.name.startswith("cam")
    )
    if variants_filter:
        wanted = set(variants_filter)
        variant_dirs = [path for path in variant_dirs if path.name in wanted]
    if not variant_dirs:
        raise FileNotFoundError(f"No cam* variant dirs under {experiment_root}")

    variants = [load_variant(path, primary_camera) for path in variant_dirs]
    shared_wide_views = check_fairness(variants, primary_camera)
    rows = variant_metric_rows(variants, primary_camera)

    calibration: dict[str, dict[str, Any]] = {}
    for variant in variants:
        summary = calibration_summary(variant, primary_camera)
        if summary is not None:
            calibration[variant.name] = summary

    write_metrics_csv(output_dir / "metrics.csv", rows)
    write_metrics_table(output_dir / "metrics_table.md", rows)

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
        "primary_camera": primary_camera,
        "fairness": {
            "shared_primary_test_views": shared_wide_views,
            "shared_primary_test_view_count": len(shared_wide_views),
        },
        "variants": rows,
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
        primary_camera=args.primary_camera,
        output_dir=output_dir,
        make_plots=not args.no_plots,
    )
    print(f"Analyzed {len(comparison['variants'])} variants -> {output_dir}")
    for row in comparison["variants"]:
        print(
            f"  {row['variant']}: PSNR={row['PSNR']:.3f} (wide {row['PSNR_wide']:.3f}) "
            f"SSIM={row['SSIM']:.4f} LPIPS={row['LPIPS']:.4f} "
            f"U_mean={row['U_mean']:.4g} observed={row['observed_anchor_fraction']:.3f}"
        )
    for warning in comparison["warnings"]:
        print(f"  WARNING: {warning}")


if __name__ == "__main__":
    main()
