#!/usr/bin/env python3

"""Select Uncertainty-Evaluation winners and create the final report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.uncertainty_evaluation import (
    directory_sha256,
    evaluation_summary,
    file_sha256,
    select_octree_candidate,
    select_uncertainty_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("octree", "uncertainty", "report"), required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def save_json(path: Path, payload: Any, *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable selection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def save_markdown_table(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_octree(root: Path) -> None:
    manifest = read_json(root / "experiment_manifest.json")
    run_index = read_json(root / "octree_runs.json")
    candidates: list[dict[str, Any]] = []
    for profile_order, profile in enumerate(run_index["profiles"]):
        for iteration in profile["iterations"]:
            summary_path = (
                root
                / "validation"
                / "octree"
                / profile["name"]
                / f"iteration_{int(iteration)}"
                / "summary.json"
            )
            summary = read_json(summary_path)
            if summary.get("split") != "validation" or summary.get("split_hash") != manifest["split_hash"]:
                raise ValueError(f"Octree candidate is not from the locked validation split: {summary_path}")
            candidates.append(
                {
                    "profile": profile["name"],
                    "profile_order": profile_order,
                    "iteration": int(iteration),
                    "model_path": profile["model_path"],
                    "summary_path": str(summary_path.resolve()),
                    "summary": summary,
                }
            )
    selected = select_octree_candidate(candidates)
    output = {
        "schema_version": 1,
        "selection_split": "validation",
        "criterion": [
            "highest primary mean PSNR",
            "lowest primary mean LPIPS",
            "highest primary mean SSIM",
            "fewer iterations",
            "declared profile order",
        ],
        "selected": {key: value for key, value in selected.items() if key != "summary"},
        "candidates": [
            {
                "profile": candidate["profile"],
                "iteration": candidate["iteration"],
                "model_path": candidate["model_path"],
                "summary_path": candidate["summary_path"],
                "primary_metrics": candidate["summary"]["groups"]["primary"]["metrics"],
            }
            for candidate in candidates
        ],
    }
    save_json(root / "octree_selection.json", output)
    save_markdown_table(
        root / "validation" / "octree_metrics.md",
        ["Profile", "Iteration", "PSNR", "SSIM", "LPIPS", "Selected"],
        [
            [
                candidate["profile"],
                candidate["iteration"],
                _fmt(candidate["summary"]["groups"]["primary"]["metrics"]["PSNR"]["mean"]),
                _fmt(candidate["summary"]["groups"]["primary"]["metrics"]["SSIM"]["mean"]),
                _fmt(candidate["summary"]["groups"]["primary"]["metrics"]["LPIPS"]["mean"]),
                "yes" if candidate is selected else "",
            ]
            for candidate in candidates
        ],
    )
    print(
        f"Selected Octree profile {selected['profile']} at iteration {selected['iteration']}"
    )


def select_uncertainty(root: Path) -> None:
    manifest = read_json(root / "experiment_manifest.json")
    octree = read_json(root / "octree_selection.json")["selected"]
    run_index = read_json(root / "uncertainty_runs.json")
    candidates: list[dict[str, Any]] = []
    reference_rgb: dict[str, dict[str, float]] | None = None
    reference_ids: list[str] | None = None
    for profile_order, profile in enumerate(run_index["profiles"]):
        summary_path = root / "validation" / "uncertainty" / profile["name"] / "summary.json"
        summary = read_json(summary_path)
        if summary.get("split") != "validation" or summary.get("split_hash") != manifest["split_hash"]:
            raise ValueError(f"Uncertainty candidate is not from the locked validation split: {summary_path}")
        per_view_path = summary_path.with_name("per_view.json")
        rows = read_json(per_view_path)["views"]
        view_ids = [str(row["view_id"]) for row in rows]
        if reference_ids is None:
            reference_ids = view_ids
            reference_rgb = {
                str(row["view_id"]): {
                    metric: float(row[metric]) for metric in ("PSNR", "SSIM", "LPIPS")
                }
                for row in rows
            }
        elif view_ids != reference_ids:
            raise ValueError(
                f"Uncertainty profiles do not contain the same ordered validation views: {per_view_path}"
            )
        assert reference_rgb is not None
        # Reuse one common RGB error vector for every uncertainty profile. The
        # selected Octree checkpoint is fixed, so only uncertainty scores vary.
        common_rows = []
        for row in rows:
            common_rows.append({**row, **reference_rgb[str(row["view_id"])]})
        summary = {**summary, **evaluation_summary(common_rows)}
        candidates.append(
            {
                "profile": profile["name"],
                "profile_order": profile_order,
                "posterior_path": profile["posterior_path"],
                "u_path": profile["u_path"],
                "summary_path": str(summary_path.resolve()),
                "summary": summary,
            }
        )
    selected = select_uncertainty_candidate(candidates)
    selection = {
        "schema_version": 1,
        "selection_split": "validation",
        "criterion": [
            "lowest primary mean normalized AUSE over PSNR/SSIM/LPIPS",
            "highest primary mean Spearman",
            "declared profile order",
        ],
        "selected": {key: value for key, value in selected.items() if key != "summary"},
        "candidates": [
            {
                "profile": candidate["profile"],
                "summary_path": candidate["summary_path"],
                "calibration": candidate["summary"]["groups"]["primary"]["calibration"],
            }
            for candidate in candidates
        ],
    }
    save_json(root / "uncertainty_selection.json", selection)
    save_markdown_table(
        root / "validation" / "uncertainty_metrics.md",
        ["Profile", "Mean normalized AUSE", "Mean Spearman", "Selected"],
        [
            [
                candidate["profile"],
                _fmt(candidate["summary"]["groups"]["primary"]["calibration"]["mean_normalized_AUSE"]),
                _fmt(candidate["summary"]["groups"]["primary"]["calibration"]["mean_spearman"]),
                "yes" if candidate is selected else "",
            ]
            for candidate in candidates
        ],
    )

    model_path = Path(octree["model_path"])
    iteration_dir = model_path / "point_cloud" / f"iteration_{int(octree['iteration'])}"
    artifact_paths = {
        "model_config": model_path / "config.yaml",
        "model_iteration": iteration_dir,
        "posterior": Path(selected["posterior_path"]),
        "uncertainty": Path(selected["u_path"]),
    }
    for name, path in artifact_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Cannot lock missing {name} artifact: {path}")
    hashes = {
        name: directory_sha256(path) if path.is_dir() else file_sha256(path)
        for name, path in artifact_paths.items()
    }
    lock = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "scene_id": manifest["scene_id"],
        "config_hash": manifest["config_hash"],
        "split_hash": manifest["split_hash"],
        "octree_profile": octree["profile"],
        "model_path": str(model_path),
        "iteration": int(octree["iteration"]),
        "uncertainty_profile": selected["profile"],
        "posterior_path": selected["posterior_path"],
        "u_path": selected["u_path"],
        "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        "artifact_hashes": hashes,
    }
    save_json(root / "selection.lock.json", lock, refuse_existing=True)
    print(f"Selected uncertainty profile {selected['profile']} and wrote immutable lock")


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def make_plots(root: Path, test: dict[str, Any], rows: list[dict[str, Any]], u_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    primary = [row for row in rows if row["is_primary"]]
    uncertainty = np.asarray([row["uncertainty_score"] for row in primary], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    transforms = {
        "PSNR": lambda row: -float(row["PSNR"]),
        "SSIM": lambda row: 1.0 - float(row["SSIM"]),
        "LPIPS": lambda row: float(row["LPIPS"]),
    }
    calibration = test["groups"]["primary"]["calibration"]
    for axis, metric in zip(axes, ("PSNR", "SSIM", "LPIPS")):
        errors = [transforms[metric](row) for row in primary]
        axis.scatter(uncertainty, errors, s=18, alpha=0.75)
        rho = calibration["metrics"].get(metric, {}).get("spearman", math.nan)
        axis.set_title(f"{metric}: rho={_fmt(rho, 3)}")
        axis.set_xlabel("alpha-normalized uncertainty")
        axis.set_ylabel("rendering error")
    fig.tight_layout()
    fig.savefig(plots / "test_calibration_scatter.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, ("PSNR", "SSIM", "LPIPS")):
        metric_payload = calibration["metrics"].get(metric)
        if metric_payload:
            axis.plot(metric_payload["fractions_removed"], metric_payload["predicted_curve"], label="uncertainty")
            axis.plot(metric_payload["fractions_removed"], metric_payload["oracle_curve"], "--", label="oracle")
        axis.set_title(f"{metric} sparsification")
        axis.set_xlabel("fraction removed")
        axis.set_ylabel("mean normalized error")
        axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "test_sparsification.png", dpi=160)
    plt.close(fig)

    values = np.load(u_path)
    finite = values[np.isfinite(values)]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(finite, bins=80)
    axis.set_yscale("log")
    axis.set_xlabel("per-anchor uncertainty U")
    axis.set_ylabel("anchor count")
    fig.tight_layout()
    fig.savefig(plots / "uncertainty_histogram.png", dpi=160)
    plt.close(fig)


def report(root: Path) -> None:
    manifest = read_json(root / "experiment_manifest.json")
    octree = read_json(root / "octree_selection.json")["selected"]
    uncertainty = read_json(root / "uncertainty_selection.json")["selected"]
    lock = read_json(root / "selection.lock.json")
    test = read_json(root / "test" / "summary.json")
    per_view = read_json(root / "test" / "per_view.json")["views"]
    make_plots(root, test, per_view, Path(lock["u_path"]))

    lines = [
        "# Uncertainty-Evaluation Report",
        "",
        f"- Dataset: `{manifest['dataset']}`",
        f"- Scene: `{manifest['scene_id']}`",
        f"- Split hash: `{manifest['split_hash']}`",
        f"- Selected Octree profile/checkpoint: `{octree['profile']}` / `{octree['iteration']}`",
        f"- Selected uncertainty profile: `{uncertainty['profile']}`",
        "",
        "## Final held-out test metrics",
        "",
        "| View group | Count | PSNR | SSIM | LPIPS | Mean normalized AUSE | Mean Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name in ("primary", "all"):
        group = test["groups"][group_name]
        calibration = group.get("calibration", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(group["view_count"]),
                    _fmt(group["metrics"]["PSNR"]["mean"]),
                    _fmt(group["metrics"]["SSIM"]["mean"]),
                    _fmt(group["metrics"]["LPIPS"]["mean"]),
                    _fmt(calibration.get("mean_normalized_AUSE")),
                    _fmt(calibration.get("mean_spearman")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Positive Spearman correlation means higher rendered uncertainty tends to identify larger held-out rendering error. Lower normalized AUSE means the uncertainty ordering more closely follows the oracle error ordering.",
            "",
            "These image-space metrics do not validate geometry. They are also sensitive to pose/calibration error, exposure, dynamic objects, and the timeline-interleaved split. VBOGS uncertainty remains blind to regions with no Octree anchor.",
            "",
            "## Diagnostics",
            "",
            "- `plots/test_calibration_scatter.png`",
            "- `plots/test_sparsification.png`",
            "- `plots/uncertainty_histogram.png`",
            "- `test/renders/`",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {root / 'report.md'}")


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    if args.phase == "octree":
        select_octree(root)
    elif args.phase == "uncertainty":
        select_uncertainty(root)
    else:
        report(root)


if __name__ == "__main__":
    main()
