#!/usr/bin/env python3

"""Compare VBOGS anchor uncertainty against projected global VBGS baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vbgs_comparison"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default="2013_05_28_drive_0007_sync")
    parser.add_argument("--comparison-root", type=Path, default=None)
    parser.add_argument("--train-bucket-root", type=Path, default=None)
    parser.add_argument("--eval-bucket-root", type=Path, default=None)
    parser.add_argument("--vbogs-u", type=Path, default=None)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def resolve_comparison_root(drive: str, comparison_root: Path | None) -> Path:
    if comparison_root is not None:
        return comparison_root.resolve()
    return (DEFAULT_OUTPUT_ROOT / drive).resolve()


def load_point_counts(bucket_root: Path) -> np.ndarray:
    with np.load(bucket_root / "pts_by_anchor.npz") as data:
        return np.asarray(data["point_counts"], dtype=np.float64).reshape(-1)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman_corr(score: np.ndarray, target: np.ndarray) -> float:
    return pearson_corr(rankdata(score), rankdata(target))


def auroc(score: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_sum_pos = float(np.sum(ranks[labels]))
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(score: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(sorted_labels.shape[0]) + 1.0)
    return float(np.sum(precision[sorted_labels]) / positives)


def top_decile_lift(score: np.ndarray, target: np.ndarray) -> float:
    if score.shape[0] == 0:
        return float("nan")
    top_k = max(1, int(np.ceil(score.shape[0] * 0.10)))
    order = np.argsort(-score, kind="mergesort")
    base = float(np.mean(target))
    if base <= 0:
        return float("nan")
    return float(np.mean(target[order[:top_k]]) / base)


def top_n_overlap(score: np.ndarray, target: np.ndarray, top_n: int) -> float:
    if score.shape[0] == 0 or top_n <= 0:
        return float("nan")
    n = min(int(top_n), int(score.shape[0]))
    score_top = set(np.argsort(-score, kind="mergesort")[:n].tolist())
    target_top = set(np.argsort(-target, kind="mergesort")[:n].tolist())
    return float(len(score_top & target_top) / n)


def evaluate_method(
    *,
    name: str,
    score: np.ndarray,
    eval_count: np.ndarray,
    eval_density: np.ndarray,
    top_n: int,
) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    finite = np.isfinite(score) & np.isfinite(eval_count) & np.isfinite(eval_density)
    score = score[finite]
    eval_count = eval_count[finite]
    eval_density = eval_density[finite]
    eval_has_points = eval_count > 0
    return {
        "method": name,
        "anchor_count": int(score.shape[0]),
        "spearman_eval_density": spearman_corr(score, eval_density),
        "auroc_eval_has_points": auroc(score, eval_has_points),
        "auprc_eval_has_points": average_precision(score, eval_has_points),
        "top_decile_eval_count_lift": top_decile_lift(score, eval_count),
        "top_n_eval_count_overlap": top_n_overlap(score, eval_count, top_n),
        "top_n": int(min(top_n, score.shape[0])) if score.shape[0] else 0,
        "score_min": float(np.min(score)) if score.shape[0] else None,
        "score_max": float(np.max(score)) if score.shape[0] else None,
    }


def load_methods(
    *,
    vbogs_u: Path,
    baseline_root: Path,
    train_count: np.ndarray,
) -> dict[str, np.ndarray]:
    methods: dict[str, np.ndarray] = {
        "vbogs": np.asarray(np.load(vbogs_u), dtype=np.float64).reshape(-1),
        "count_baseline": 1.0 / np.sqrt(train_count.astype(np.float64) + 1.0),
    }
    for child in sorted(baseline_root.glob("K_*")):
        u_path = child / "U_baseline.npy"
        if not u_path.exists():
            continue
        methods[f"global_vbgs_{child.name}"] = np.asarray(
            np.load(u_path),
            dtype=np.float64,
        ).reshape(-1)
    return methods


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "anchor_count",
        "spearman_eval_density",
        "auroc_eval_has_points",
        "auprc_eval_has_points",
        "top_decile_eval_count_lift",
        "top_n_eval_count_overlap",
        "top_n",
        "score_min",
        "score_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["spearman_eval_density"]
            if np.isfinite(row["spearman_eval_density"])
            else -np.inf
        ),
        reverse=True,
    )
    lines = [
        "# VBGS vs VBOGS Uncertainty Comparison",
        "",
        "| Method | Spearman eval density | AUROC eval has points | Top decile lift |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {method} | {spearman:.4f} | {auroc:.4f} | {lift:.4f} |".format(
                method=row["method"],
                spearman=row["spearman_eval_density"],
                auroc=row["auroc_eval_has_points"],
                lift=row["top_decile_eval_count_lift"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(plots_dir: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib import pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"enabled": False, "reason": str(exc)}

    methods = [row["method"] for row in rows]
    spearman = [row["spearman_eval_density"] for row in rows]
    lift = [row["top_decile_eval_count_lift"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.4), 4))
    ax.bar(methods, spearman)
    ax.set_ylabel("Spearman vs eval density")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    spearman_path = plots_dir / "spearman_eval_density.png"
    fig.savefig(spearman_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.4), 4))
    ax.bar(methods, lift)
    ax.set_ylabel("Top decile eval-count lift")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    lift_path = plots_dir / "top_decile_lift.png"
    fig.savefig(lift_path)
    plt.close(fig)

    return {
        "enabled": True,
        "paths": [str(spearman_path), str(lift_path)],
    }


def run_comparison(
    *,
    drive: str,
    comparison_root: Path,
    train_bucket_root: Path,
    eval_bucket_root: Path,
    vbogs_u: Path,
    baseline_root: Path,
    top_n: int,
    no_plots: bool,
) -> dict[str, Any]:
    train_count = load_point_counts(train_bucket_root)
    eval_count = load_point_counts(eval_bucket_root)
    if train_count.shape != eval_count.shape:
        raise ValueError(
            f"Train/eval anchor counts differ: {train_count.shape} vs {eval_count.shape}"
        )

    eval_density = eval_count / np.maximum(train_count, 1.0)
    methods = load_methods(
        vbogs_u=vbogs_u,
        baseline_root=baseline_root,
        train_count=train_count,
    )
    rows = [
        evaluate_method(
            name=name,
            score=score,
            eval_count=eval_count,
            eval_density=eval_density,
            top_n=top_n,
        )
        for name, score in methods.items()
    ]

    metrics_path = comparison_root / "comparison_metrics.json"
    csv_path = comparison_root / "comparison_metrics.csv"
    summary_path = comparison_root / "comparison_summary.md"
    plots_dir = comparison_root / "plots"

    plot_metadata = {"enabled": False, "reason": "disabled"}
    if not no_plots:
        plot_metadata = write_plots(plots_dir, rows)
    else:
        plots_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "drive": drive,
        "comparison_root": str(comparison_root),
        "train_bucket_root": str(train_bucket_root),
        "eval_bucket_root": str(eval_bucket_root),
        "vbogs_u": str(vbogs_u),
        "baseline_root": str(baseline_root),
        "anchor_count": int(train_count.shape[0]),
        "eval_positive_anchor_count": int(np.count_nonzero(eval_count > 0)),
        "top_n": int(top_n),
        "methods": rows,
        "plots": plot_metadata,
    }
    save_json(metrics_path, payload)
    write_csv(csv_path, rows)
    write_summary(summary_path, rows)
    save_json(plots_dir / "plot_metadata.json", plot_metadata)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    comparison_root = resolve_comparison_root(args.drive, args.comparison_root)
    train_bucket_root = (
        args.train_bucket_root or comparison_root / "m4_train"
    ).resolve()
    eval_bucket_root = (args.eval_bucket_root or comparison_root / "m4_eval").resolve()
    vbogs_u = (args.vbogs_u or train_bucket_root / "U.npy").resolve()
    baseline_root = (args.baseline_root or comparison_root / "vbgs_global").resolve()
    payload = run_comparison(
        drive=args.drive,
        comparison_root=comparison_root,
        train_bucket_root=train_bucket_root,
        eval_bucket_root=eval_bucket_root,
        vbogs_u=vbogs_u,
        baseline_root=baseline_root,
        top_n=args.top_n,
        no_plots=args.no_plots,
    )
    print(f"Wrote {comparison_root / 'comparison_metrics.json'}")
    print(f"Wrote {comparison_root / 'comparison_metrics.csv'}")
    print(f"Wrote {comparison_root / 'comparison_summary.md'}")
    print(f"Compared {len(payload['methods'])} method(s).")


if __name__ == "__main__":
    main()
