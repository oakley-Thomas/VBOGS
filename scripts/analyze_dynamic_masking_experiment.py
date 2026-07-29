#!/usr/bin/env python3
"""Compare fixed paired unmasked and dynamic-masked experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from vbogs.uncertainty_evaluation import calibration_summary, split_fingerprint


VARIANTS = ("unmasked", "dynamic_masked")
METRICS = ("PSNR", "SSIM", "LPIPS", "PSNR_static", "SSIM_static", "LPIPS_static")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--dynamic-mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def variant_dir(root: Path, name: str, scene: str) -> Path:
    path = root / name / scene
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {name} run directory: {path}")
    return path


def model_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest["source_paths"]["octree_model_path"])
    size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0
    return {"model_path": str(path), "checkpoint_bytes": int(size)}


def load_variant(root: Path, name: str, scene: str) -> dict[str, Any]:
    run_dir = variant_dir(root, name, scene)
    manifest = read_json(run_dir / "run_manifest.json")
    prepared = read_json(run_dir / "prepared" / "metadata.json")
    evaluation_dir = run_dir / "evaluation" / "static_test"
    per_view_payload = read_json(evaluation_dir / "per_view.json")
    summary = read_json(evaluation_dir / "summary.json")
    uncertainty_path = run_dir / "uncertainty" / "U.npy"
    uncertainty = np.load(uncertainty_path) if uncertainty_path.exists() else np.empty(0)
    points = read_json(run_dir / "pointclouds" / "stereo" / "points_world_metadata.json")
    bucket_root = Path(manifest["source_paths"]["bucket_root"])
    bucket_metadata = read_json(bucket_root / "bucket_metadata.json") if (bucket_root / "bucket_metadata.json").exists() else {}
    fit_metadata = read_json(bucket_root / "fit_metadata.json") if (bucket_root / "fit_metadata.json").exists() else {}
    uncertainty_metadata = read_json(run_dir / "uncertainty" / "uncertainty_metadata.json")
    return {
        "name": name,
        "run_dir": run_dir,
        "manifest": manifest,
        "prepared": prepared,
        "per_view": per_view_payload,
        "summary": summary,
        "points": points,
        "anchors": {
            "anchor_count": bucket_metadata.get("anchor_count", uncertainty_metadata.get("anchor_count")),
            "nonempty_anchor_count": bucket_metadata.get("nonempty_anchor_count"),
            "levels": bucket_metadata.get("levels"),
            "assignment_count_by_level": bucket_metadata.get("assignment_count_by_level"),
            "fit_elapsed_sec": fit_metadata.get("elapsed_sec"),
        },
        "model": model_summary(manifest),
        "uncertainty": {
            "count": int(uncertainty.size),
            "mean": float(np.mean(uncertainty)) if uncertainty.size else math.nan,
            "median": float(np.median(uncertainty)) if uncertainty.size else math.nan,
            "p90": float(np.percentile(uncertainty, 90)) if uncertainty.size else math.nan,
        },
    }


def scrub_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model_params"]
    model.pop("source_path", None)
    model.pop("dataset_name", None)
    model.pop("scene_name", None)
    model.pop("add_mask", None)
    return config


def validate_pair(left: dict[str, Any], right: dict[str, Any], dynamic_mask_root: Path) -> None:
    left_split = split_fingerprint(left["prepared"])
    right_split = split_fingerprint(right["prepared"])
    if left_split != right_split:
        raise ValueError("Paired variants have different selected frames/cameras/splits")
    if left["per_view"]["split_hash"] != right["per_view"]["split_hash"]:
        raise ValueError("Evaluator split hashes differ between variants")
    if left["per_view"].get("image_mask_sha256") != right["per_view"].get("image_mask_sha256"):
        raise ValueError("Variants were not evaluated with the same dynamic-mask artifact")
    expected = str(dynamic_mask_root.resolve())
    if left["per_view"].get("image_mask_root") != expected:
        raise ValueError("Evaluator mask root does not match --dynamic-mask-root")
    configs = [scrub_config(Path(item["manifest"]["source_paths"]["octree_model_path"]) / "config.yaml") for item in (left, right)]
    if configs[0] != configs[1]:
        raise ValueError("Octree configs differ beyond source/output paths and add_mask")
    for item, expected_add_mask in ((left, False), (right, True)):
        import yaml
        with (Path(item["manifest"]["source_paths"]["octree_model_path"]) / "config.yaml").open("r", encoding="utf-8") as handle:
            actual = bool(yaml.safe_load(handle)["model_params"].get("add_mask"))
        if actual != expected_add_mask:
            raise ValueError(f"{item['name']} add_mask={actual}, expected {expected_add_mask}")


def paired_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_rows = {row["view_id"]: row for row in left["per_view"]["views"]}
    right_rows = {row["view_id"]: row for row in right["per_view"]["views"]}
    if set(left_rows) != set(right_rows):
        raise ValueError("Evaluator view IDs differ between paired variants")
    rows = []
    for view_id in sorted(left_rows):
        before, after = left_rows[view_id], right_rows[view_id]
        row: dict[str, Any] = {
            "view_id": view_id,
            "frame_id": before["frame_id"],
            "camera_id": before["camera_id"],
            "is_primary": before["is_primary"],
            "static_pixel_fraction": before.get("static_pixel_fraction"),
        }
        for metric in METRICS:
            row[f"unmasked_{metric}"] = before.get(metric)
            row[f"dynamic_masked_{metric}"] = after.get(metric)
            if before.get(metric) is not None and after.get(metric) is not None:
                row[f"delta_{metric}"] = float(after[metric]) - float(before[metric])
        row["unmasked_uncertainty_score_static"] = before.get("uncertainty_score_static")
        row["dynamic_masked_uncertainty_score_static"] = after.get("uncertainty_score_static")
        rows.append(row)
    return rows


def summary_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("primary", "all"):
        left_group = left["summary"]["groups"][group]
        right_group = right["summary"]["groups"][group]
        metrics: dict[str, Any] = {}
        for key, source in (("full", "metrics"), ("static", "static_metrics")):
            for metric, values in left_group.get(source, {}).items():
                if metric not in right_group.get(source, {}):
                    continue
                metrics[metric] = {
                    "unmasked_mean": values["mean"],
                    "dynamic_masked_mean": right_group[source][metric]["mean"],
                    "delta": right_group[source][metric]["mean"] - values["mean"],
                }
        result[group] = {"metrics": metrics}
        for calibration_key in ("calibration", "calibration_static"):
            if calibration_key in left_group and calibration_key in right_group:
                result[group][calibration_key] = {
                    "unmasked": left_group[calibration_key],
                    "dynamic_masked": right_group[calibration_key],
                }
    return result


def copy_visual_review(rows: list[dict[str, Any]], left: dict[str, Any], right: dict[str, Any], dynamic_mask_root: Path, output_dir: Path) -> list[dict[str, str]]:
    primary = [row for row in rows if row["is_primary"]]
    if not primary:
        return []
    indices = np.linspace(0, len(primary) - 1, min(5, len(primary)), dtype=int)
    selected = [primary[index] for index in indices]
    result = []
    for ordinal, row in enumerate(selected):
        name = Path(row["view_id"])
        safe = str(name.with_suffix("")).replace("/", "__")
        index = next(item["index"] for item in left["per_view"]["views"] if item["view_id"] == row["view_id"])
        rendered_name = f"{int(index):05d}_{safe}.png"
        dest = output_dir / f"{ordinal:02d}_{safe}"
        dest.mkdir(parents=True, exist_ok=True)
        sources = {
            "ground_truth.png": left["run_dir"] / "evaluation/static_test/renders/ground_truth" / rendered_name,
            "unmasked_render.png": left["run_dir"] / "evaluation/static_test/renders/rgb" / rendered_name,
            "unmasked_error.png": left["run_dir"] / "evaluation/static_test/renders/absolute_error" / rendered_name,
            "unmasked_uncertainty.png": left["run_dir"] / "evaluation/static_test/renders/uncertainty" / rendered_name,
            "dynamic_masked_render.png": right["run_dir"] / "evaluation/static_test/renders/rgb" / rendered_name,
            "dynamic_masked_error.png": right["run_dir"] / "evaluation/static_test/renders/absolute_error" / rendered_name,
            "dynamic_masked_uncertainty.png": right["run_dir"] / "evaluation/static_test/renders/uncertainty" / rendered_name,
            "dynamic_mask_overlay.png": dynamic_mask_root / "overlays" / name,
        }
        for target, source in sources.items():
            if source.exists():
                shutil.copy2(source, dest / target)
        result.append({"view_id": row["view_id"], "directory": str(dest)})
    return result


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    output = (args.output_dir or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    dynamic_root = args.dynamic_mask_root.resolve()
    left, right = (load_variant(root, variant, args.scene_id) for variant in VARIANTS)
    validate_pair(left, right, dynamic_root)
    rows = paired_rows(left, right)
    review = copy_visual_review(rows, left, right, dynamic_root, output / "visual_review")
    payload = {
        "schema_version": 1,
        "scene_id": args.scene_id,
        "variants": {
            item["name"]: {
                "points": item["points"], "model": item["model"], "uncertainty": item["uncertainty"],
                "anchors": item["anchors"],
                "static_calibration": item["summary"]["groups"]["primary"].get("calibration_static"),
            }
            for item in (left, right)
        },
        "summary": summary_delta(left, right),
        "per_view": rows,
        "visual_review": review,
    }
    (output / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    with (output / "per_view_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Dynamic-masking paired comparison", "", "Static-region metrics are primary; deltas are dynamic-masked minus unmasked.", ""]
    for group, value in payload["summary"].items():
        lines.extend([f"## {group}", "", "| Metric | Unmasked | Dynamic masked | Delta |", "|---|---:|---:|---:|"])
        for metric, entry in value["metrics"].items():
            lines.append(f"| {metric} | {entry['unmasked_mean']:.6g} | {entry['dynamic_masked_mean']:.6g} | {entry['delta']:+.6g} |")
        lines.append("")
    (output / "metrics_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote paired comparison to {output}")


if __name__ == "__main__":
    main()
