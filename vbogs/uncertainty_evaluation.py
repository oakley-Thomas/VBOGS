"""Pure helpers for the Uncertainty-Evaluation experiment.

The GPU renderer lives in ``scripts/evaluate_uncertainty_views.py``.  This
module intentionally contains the split, metric, selection, and provenance
logic that can be tested without Torch, JAX, or the upstream submodules.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


METRICS = ("PSNR", "SSIM", "LPIPS")
STATIC_METRICS = tuple(f"{metric}_static" for metric in METRICS)
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class EvaluationView:
    view_id: str
    image_name: str
    camera_id: str
    frame_id: int
    split: str
    is_primary: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash file names and contents below ``path`` in stable order."""

    root = Path(path)
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = child.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _split_frames(metadata: dict[str, Any]) -> dict[str, list[int]]:
    raw = metadata.get("frame_splits", {})
    if not isinstance(raw, dict):
        raise ValueError("metadata.frame_splits must be an object")
    result = {name: [int(value) for value in raw.get(name, [])] for name in SPLITS}
    return result


def validate_split_integrity(metadata: dict[str, Any]) -> dict[str, list[int]]:
    frame_splits = _split_frames(metadata)
    seen: dict[int, str] = {}
    for split, frame_ids in frame_splits.items():
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f"Duplicate frame id inside {split} split")
        for frame_id in frame_ids:
            if frame_id in seen:
                raise ValueError(
                    f"Frame {frame_id} occurs in both {seen[frame_id]} and {split} splits"
                )
            seen[frame_id] = split

    records = metadata.get("frame_records", [])
    if not isinstance(records, list):
        raise ValueError("metadata.frame_records must be a list")
    record_lookup: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every frame record must be an object")
        frame_id = int(record["frame_id"])
        split = str(record["split"])
        if split not in SPLITS:
            raise ValueError(f"Unsupported split {split!r} for frame {frame_id}")
        if frame_id in record_lookup:
            raise ValueError(f"Duplicate frame record for frame {frame_id}")
        record_lookup[frame_id] = split
        if seen.get(frame_id) != split:
            raise ValueError(
                f"Frame record {frame_id} says {split}, frame_splits says {seen.get(frame_id)!r}"
            )
    if set(record_lookup) != set(seen):
        missing = sorted(set(seen) - set(record_lookup))
        extra = sorted(set(record_lookup) - set(seen))
        raise ValueError(f"frame_records/frame_splits mismatch; missing={missing}, extra={extra}")
    return frame_splits


def split_fingerprint(metadata: dict[str, Any]) -> str:
    validate_split_integrity(metadata)
    records = []
    for record in metadata["frame_records"]:
        cameras: list[tuple[str, str]] = []
        if metadata.get("dataset") == "nvidia_ncore":
            for camera_id, payload in sorted(record.get("cameras", {}).items()):
                cameras.append((str(camera_id), str(payload["image_name"])))
        else:
            for payload in record.get("images", []):
                cameras.append((str(payload["camera"]), str(payload["image_name"])))
        records.append(
            {
                "frame_id": int(record["frame_id"]),
                "split": str(record["split"]),
                "cameras": cameras,
            }
        )
    return canonical_json_hash({"frame_splits": _split_frames(metadata), "records": records})


def primary_camera_id(metadata: dict[str, Any]) -> str:
    if metadata.get("dataset") == "nvidia_ncore":
        value = metadata.get("primary_camera_id")
        if not value:
            raise ValueError("NCore metadata is missing primary_camera_id")
        return str(value)
    return "image_00"


def expected_training_image_count(metadata: dict[str, Any]) -> int:
    count = 0
    for record in metadata.get("frame_records", []):
        if record.get("split") != "train":
            continue
        if metadata.get("dataset") == "nvidia_ncore":
            count += len(record.get("cameras", {}))
        else:
            count += len(record.get("images", []))
    return count


def evaluation_views(metadata: dict[str, Any], split: str) -> list[EvaluationView]:
    if split not in ("validation", "test"):
        raise ValueError("Held-out evaluation only supports validation or test")
    validate_split_integrity(metadata)
    primary = primary_camera_id(metadata)
    views: list[EvaluationView] = []
    for record in metadata["frame_records"]:
        if record["split"] != split:
            continue
        frame_id = int(record["frame_id"])
        if metadata.get("dataset") == "nvidia_ncore":
            camera_payloads = record.get("cameras", {})
            order = metadata.get("camera_ids", sorted(camera_payloads))
            entries = [
                (str(camera_id), camera_payloads[camera_id])
                for camera_id in order
                if camera_id in camera_payloads
            ]
        else:
            entries = [
                (str(payload["camera"]), payload)
                for payload in record.get("images", [])
            ]
        for camera_id, payload in entries:
            image_name = str(payload["image_name"])
            views.append(
                EvaluationView(
                    view_id=image_name,
                    image_name=image_name,
                    camera_id=camera_id,
                    frame_id=frame_id,
                    split=split,
                    is_primary=camera_id == primary,
                )
            )
    if not views:
        raise ValueError(f"No {split} views found in prepared metadata")
    if not any(view.is_primary for view in views):
        raise ValueError(f"No primary-camera {split} views found for {primary!r}")
    identifiers = [view.view_id for view in views]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Image names are not unique in the {split} split")
    return views


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": math.nan, "median": math.nan, "p10": math.nan, "p90": math.nan}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
    }


def psnr_unit_range(render: np.ndarray, ground_truth: np.ndarray) -> float:
    """Reference PSNR for RGB arrays in [0, 1]."""

    rendered = np.asarray(render, dtype=np.float64)
    target = np.asarray(ground_truth, dtype=np.float64)
    if rendered.shape != target.shape:
        raise ValueError("PSNR arrays must have the same shape")
    mse = float(np.mean((rendered - target) ** 2))
    return math.inf if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 1:
        raise ValueError("Spearman inputs must be equally sized 1D arrays")
    if x_arr.size < 2:
        return math.nan

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=np.float64)
        index = 0
        while index < values.size:
            end = index
            while end + 1 < values.size and values[order[end + 1]] == values[order[index]]:
                end += 1
            result[order[index : end + 1]] = 0.5 * (index + end) + 1.0
            index = end + 1
        return result

    rx = ranks(x_arr)
    ry = ranks(y_arr)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return math.nan if denominator == 0.0 else float(np.sum(rx * ry) / denominator)


def sparsification_curve(
    uncertainty: Sequence[float], errors: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uncertainty_arr = np.asarray(uncertainty, dtype=np.float64)
    error_arr = np.asarray(errors, dtype=np.float64)
    if uncertainty_arr.shape != error_arr.shape or error_arr.ndim != 1:
        raise ValueError("Sparsification inputs must be equally sized 1D arrays")
    if error_arr.size == 0:
        raise ValueError("Sparsification requires at least one view")

    def curve(order: np.ndarray) -> np.ndarray:
        retained = error_arr[order[::-1]]
        return (np.cumsum(retained) / np.arange(1, retained.size + 1))[::-1]

    predicted = curve(np.argsort(-uncertainty_arr, kind="mergesort"))
    oracle = curve(np.argsort(-error_arr, kind="mergesort"))
    fractions = np.arange(error_arr.size, dtype=np.float64) / error_arr.size
    return fractions, predicted, oracle


def ause(predicted: Sequence[float], oracle: Sequence[float]) -> float:
    return float(np.mean(np.asarray(predicted, dtype=np.float64) - np.asarray(oracle, dtype=np.float64)))


def normalized_errors(rows: Sequence[dict[str, Any]], metric: str) -> tuple[np.ndarray | None, str | None]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    base_metric = metric.removesuffix("_static")
    if values.size < 2:
        return None, f"{metric} has fewer than two finite views; omitted from calibration"
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if not np.isfinite(span) or span <= 0.0:
        return None, f"{metric} is constant on the selected views; omitted from calibration"
    if base_metric in ("PSNR", "SSIM"):
        return (high - values) / span, None
    if base_metric == "LPIPS":
        return (values - low) / span, None
    raise ValueError(f"Unsupported metric: {metric}")


def view_score_fields(
    uncertainty_sum: float,
    observed_uncertainty_sum: float,
    observed_sum: float,
    alpha_sum: float,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """CPU-testable observed-only uncertainty score fields for one render."""

    return {
        "observed_sum": float(observed_sum),
        "observed_uncertainty_sum": float(observed_uncertainty_sum),
        "uncertainty_score_observed": float(observed_uncertainty_sum / (observed_sum + eps)),
        "unobserved_fraction": float(np.clip(1.0 - observed_sum / (alpha_sum + eps), 0.0, 1.0)),
    }


def calibration_summary(
    rows: Sequence[dict[str, Any]],
    score_key: str = "uncertainty_score",
    metric_keys: Sequence[str] = METRICS,
) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "metrics": {}, "mean_normalized_AUSE": math.nan, "mean_spearman": math.nan, "warnings": ["No views"]}
    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    for metric in metric_keys:
        metric_rows = [
            row
            for row in rows
            if row.get(score_key) is not None
            and row.get(metric) is not None
            and np.isfinite(float(row[score_key]))
            and np.isfinite(float(row[metric]))
        ]
        uncertainty = np.asarray([float(row[score_key]) for row in metric_rows], dtype=np.float64)
        errors, warning = normalized_errors(metric_rows, metric)
        if warning:
            warnings.append(warning)
            continue
        assert errors is not None
        fractions, predicted, oracle = sparsification_curve(uncertainty, errors)
        metrics[metric] = {
            "spearman": spearman(uncertainty, errors),
            "normalized_AUSE": ause(predicted, oracle),
            "fractions_removed": fractions.tolist(),
            "predicted_curve": predicted.tolist(),
            "oracle_curve": oracle.tolist(),
        }
    ause_values = [entry["normalized_AUSE"] for entry in metrics.values()]
    rho_values = [entry["spearman"] for entry in metrics.values() if np.isfinite(entry["spearman"])]
    return {
        "count": len(rows),
        "metrics": metrics,
        "mean_normalized_AUSE": float(np.mean(ause_values)) if ause_values else math.nan,
        "mean_spearman": float(np.mean(rho_values)) if rho_values else math.nan,
        "warnings": warnings,
    }


def evaluation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "primary": [row for row in rows if bool(row["is_primary"])],
        "all": list(rows),
    }
    result: dict[str, Any] = {"view_count": len(rows), "groups": {}}
    for name, group_rows in groups.items():
        metrics = {
            metric: summarize_values(float(row[metric]) for row in group_rows)
            for metric in METRICS
        }
        payload: dict[str, Any] = {"view_count": len(group_rows), "metrics": metrics}
        if group_rows and any(row.get("PSNR_static") is not None for row in group_rows):
            payload["static_metrics"] = {
                metric: summarize_values(
                    float(row[metric])
                    for row in group_rows
                    if row.get(metric) is not None
                )
                for metric in STATIC_METRICS
            }
            payload["static_pixel_fraction"] = summarize_values(
                float(row["static_pixel_fraction"])
                for row in group_rows
                if row.get("static_pixel_fraction") is not None
            )
        if group_rows and all(row.get("uncertainty_score") is not None for row in group_rows):
            payload["uncertainty_score"] = summarize_values(
                float(row["uncertainty_score"]) for row in group_rows
            )
            payload["calibration"] = calibration_summary(group_rows)
        if group_rows and all(row.get("uncertainty_score_observed") is not None for row in group_rows):
            payload["calibration_observed"] = calibration_summary(
                group_rows, score_key="uncertainty_score_observed"
            )
        if group_rows and all(row.get("uncertainty_score_static") is not None for row in group_rows):
            payload["calibration_static"] = calibration_summary(
                group_rows,
                score_key="uncertainty_score_static",
                metric_keys=STATIC_METRICS,
            )
        if group_rows and all(row.get("frame_id") is not None for row in group_rows):
            payload["frame_id_null"] = calibration_summary(group_rows, score_key="frame_id")
        if group_rows and all(row.get("unobserved_fraction") is not None for row in group_rows):
            payload["unobserved_fraction"] = summarize_values(
                float(row["unobserved_fraction"]) for row in group_rows
            )
        result["groups"][name] = payload
    return result


def select_octree_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No Octree validation candidates were provided")

    def key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        metrics = candidate["summary"]["groups"]["primary"]["metrics"]
        return (
            -float(metrics["PSNR"]["mean"]),
            float(metrics["LPIPS"]["mean"]),
            -float(metrics["SSIM"]["mean"]),
            int(candidate["iteration"]),
            int(candidate["profile_order"]),
        )

    return min(candidates, key=key)


def select_uncertainty_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No uncertainty validation candidates were provided")

    def key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        primary = candidate["summary"]["groups"]["primary"]
        calibration = primary.get("calibration_observed") or primary.get("calibration", {})
        objective = float(calibration.get("mean_normalized_AUSE", math.nan))
        correlation = float(calibration.get("mean_spearman", math.nan))
        if not np.isfinite(objective):
            objective = math.inf
        if not np.isfinite(correlation):
            correlation = -math.inf
        return objective, -correlation, int(candidate["profile_order"])

    selected = min(candidates, key=key)
    primary = selected["summary"]["groups"]["primary"]
    calibration = primary.get("calibration_observed") or primary.get("calibration", {})
    objective = calibration["mean_normalized_AUSE"]
    if not np.isfinite(objective):
        raise ValueError("Every uncertainty candidate has an invalid calibration objective")
    return selected


def verify_selection_lock(
    lock: dict[str, Any],
    *,
    config_hash: str,
    split_hash: str,
    artifact_paths: dict[str, Path],
) -> None:
    if lock.get("config_hash") != config_hash:
        raise ValueError("Selection lock config hash does not match the active experiment config")
    if lock.get("split_hash") != split_hash:
        raise ValueError("Selection lock split hash does not match prepared metadata")
    expected = lock.get("artifact_hashes", {})
    for name, path in artifact_paths.items():
        actual = directory_sha256(path) if Path(path).is_dir() else file_sha256(path)
        if expected.get(name) != actual:
            raise ValueError(f"Selection lock hash mismatch for {name}: {path}")
