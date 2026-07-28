from pathlib import Path

from vbogs.web.store import RunStore, utc_now


def make_record(root: Path, run_id: str = "run-123456789abc") -> dict:
    return {
        "id": run_id,
        "owner": "operator@example.test",
        "dataset": "kitti360",
        "scene_id": "drive",
        "preset": "kitti360-dev",
        "start_at": "prepare",
        "stop_after": "bundle",
        "created_at": utc_now(),
        "config_path": str(root / "config.yaml"),
        "workspace_path": str(root / "workspace"),
        "output_path": str(root / "output"),
        "command": ["scripts/run_pipeline.sh"],
    }


def test_store_persists_queue_events_and_resume(tmp_path):
    store = RunStore(tmp_path / "control.sqlite3")
    store.create_run(make_record(tmp_path))
    assert store.queued_runs()[0]["id"] == "run-123456789abc"
    store.transition("run-123456789abc", "running", gpu_id="0")
    store.transition("run-123456789abc", "failed", error="test failure")
    resumed = store.requeue("run-123456789abc", start_at="bucket", stop_after="bundle")
    assert resumed is not None
    assert resumed["status"] == "queued"
    assert resumed["start_at"] == "bucket"
    assert any(event["type"] == "requeued" for event in store.events("run-123456789abc"))


def test_store_marks_active_runs_interrupted_after_restart(tmp_path):
    store = RunStore(tmp_path / "control.sqlite3")
    store.create_run(make_record(tmp_path))
    store.transition("run-123456789abc", "running", gpu_id="0")
    assert store.mark_active_interrupted() == 1
    assert store.get_run("run-123456789abc")["status"] == "interrupted"


def test_store_lists_only_requested_statuses(tmp_path):
    store = RunStore(tmp_path / "control.sqlite3")
    store.create_run(make_record(tmp_path, "run-active"))
    store.create_run(make_record(tmp_path, "run-completed"))
    store.transition("run-completed", "completed")

    assert [run["id"] for run in store.list_runs(statuses=("queued", "starting", "running", "cancelling"))] == ["run-active"]
    assert [run["id"] for run in store.list_runs(statuses=("completed",))] == ["run-completed"]
