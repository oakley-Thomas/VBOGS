"""Safe progress projection for GUI pipeline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PIPELINE_STAGES = (
    "dynamic-mask", "prepare", "train", "stereo", "bucket", "fit", "inspect",
    "uncertainty", "map-viz", "render", "nbv", "nbv-viz", "bundle",
)


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and isinstance(value.get("type"), str):
                    result.append(value)
    except OSError:
        return []
    return result


def _training_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    current, total = value.get("current_iterations"), value.get("total_iterations")
    if (
        not isinstance(current, int) or isinstance(current, bool)
        or not isinstance(total, int) or isinstance(total, bool)
        or total <= 0
        or current < 0
        or current > total
        or value.get("state") not in {"starting", "running", "finalizing", "completed", "failed"}
    ):
        return None
    result = {
        "state": value["state"],
        "current_iterations": current,
        "total_iterations": total,
    }
    if isinstance(value.get("updated_at"), str):
        result["updated_at"] = value["updated_at"]
    return result


def _fallback_stages(run: dict[str, Any]) -> list[str]:
    try:
        first = PIPELINE_STAGES.index(str(run["start_at"]))
        last = PIPELINE_STAGES.index(str(run["stop_after"]))
    except (KeyError, ValueError):
        return []
    return list(PIPELINE_STAGES[first : last + 1]) if first <= last else []


def project_run_progress(run: dict[str, Any]) -> dict[str, Any]:
    """Combine lifecycle events and a train snapshot into a browser-safe payload."""

    workspace = Path(str(run["workspace_path"]))
    events = _events(workspace / "pipeline.events.jsonl")
    last_started = max((index for index, event in enumerate(events) if event["type"] == "run_started"), default=-1)
    active_events = events[last_started:] if last_started >= 0 else []
    declared_stages = active_events[0].get("stages") if active_events else None
    stages = [stage for stage in declared_stages if isinstance(stage, str)] if isinstance(declared_stages, list) else _fallback_stages(run)
    completed = {
        str(event.get("stage"))
        for event in active_events
        if event["type"] == "stage_completed" and str(event.get("stage")) in stages
    }
    current_stage: str | None = None
    for event in active_events:
        stage = event.get("stage")
        if event["type"] == "stage_started" and stage in stages:
            current_stage = str(stage)
        elif event["type"] == "stage_completed" and stage == current_stage:
            current_stage = None

    status = str(run.get("status", "queued"))
    total_stages = len(stages)
    completed_stages = len(completed)
    if status == "completed":
        completed_stages = total_stages
        current_stage = None

    training = _training_snapshot(workspace / "training_progress.json") if current_stage == "train" else None
    active_fraction = 0.0
    if training is not None:
        active_fraction = min(training["current_iterations"] / training["total_iterations"], 0.99)
    if status == "completed":
        fraction = 1.0
    elif total_stages:
        fraction = min(0.99, (completed_stages + active_fraction) / total_stages)
    else:
        fraction = 0.0

    current: dict[str, Any] | None = None
    if current_stage is not None and current_stage in stages:
        current = {"name": current_stage, "index": stages.index(current_stage) + 1, "total": total_stages}
    display_state = training["state"] if training and training["state"] == "finalizing" else status
    return {
        "state": display_state,
        "status": status,
        "overall": {
            "completed_stages": completed_stages,
            "total_stages": total_stages,
            "percent": round(fraction * 100, 1),
        },
        "current_stage": current,
        "training": training,
    }
