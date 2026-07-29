import json
from pathlib import Path

from vbogs.web.progress import project_run_progress


def make_run(tmp_path: Path, *, status: str = "running", start_at: str = "prepare", stop_after: str = "bundle") -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {"status": status, "start_at": start_at, "stop_after": stop_after, "workspace_path": str(workspace)}


def write_events(run: dict, events: list[dict]) -> None:
    Path(run["workspace_path"], "pipeline.events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def write_training(run: dict, *, state: str = "running", current: int = 4200, total: int = 30000) -> None:
    Path(run["workspace_path"], "training_progress.json").write_text(
        json.dumps({"state": state, "current_iterations": current, "total_iterations": total, "updated_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )


def test_progress_uses_declared_stages_and_live_training_fraction(tmp_path):
    run = make_run(tmp_path)
    write_events(run, [
        {"type": "run_started", "stages": ["prepare", "train", "fit"]},
        {"type": "stage_started", "stage": "prepare"},
        {"type": "stage_completed", "stage": "prepare"},
        {"type": "stage_started", "stage": "train"},
    ])
    write_training(run)

    progress = project_run_progress(run)

    assert progress["current_stage"] == {"name": "train", "index": 2, "total": 3}
    assert progress["overall"] == {"completed_stages": 1, "total_stages": 3, "percent": 38.0}
    assert progress["training"] == {
        "state": "running", "current_iterations": 4200, "total_iterations": 30000,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def test_progress_holds_overall_below_complete_while_training_finalizes(tmp_path):
    run = make_run(tmp_path, start_at="train", stop_after="train")
    write_events(run, [{"type": "run_started", "stages": ["train"]}, {"type": "stage_started", "stage": "train"}])
    write_training(run, state="finalizing", current=30000, total=30000)

    progress = project_run_progress(run)

    assert progress["state"] == "finalizing"
    assert progress["overall"]["percent"] == 99.0


def test_progress_uses_only_latest_pipeline_attempt_after_resume(tmp_path):
    run = make_run(tmp_path, start_at="train", stop_after="train")
    write_events(run, [
        {"type": "run_started", "stages": ["prepare", "train"]},
        {"type": "stage_started", "stage": "prepare"},
        {"type": "stage_completed", "stage": "prepare"},
        {"type": "run_started", "stages": ["train"]},
        {"type": "stage_started", "stage": "train"},
    ])
    write_training(run, current=3000, total=30000)

    progress = project_run_progress(run)

    assert progress["overall"] == {"completed_stages": 0, "total_stages": 1, "percent": 10.0}


def test_progress_handles_corrupt_snapshots_and_terminal_states(tmp_path):
    run = make_run(tmp_path, status="failed", start_at="prepare", stop_after="train")
    write_events(run, [
        {"type": "run_started", "stages": ["prepare", "train"]},
        {"type": "stage_started", "stage": "prepare"},
        {"type": "stage_completed", "stage": "prepare"},
        {"type": "stage_started", "stage": "train"},
    ])
    Path(run["workspace_path"], "training_progress.json").write_text("{partial", encoding="utf-8")

    failed = project_run_progress(run)
    assert failed["training"] is None
    assert failed["overall"]["percent"] == 50.0

    run["status"] = "completed"
    completed = project_run_progress(run)
    assert completed["current_stage"] is None
    assert completed["overall"] == {"completed_stages": 2, "total_stages": 2, "percent": 100.0}
