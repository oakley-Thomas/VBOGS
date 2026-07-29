from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import vbogs.web.app as web_app


REPO_ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"X-Forwarded-User": "operator@example.test"}


def make_app(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data" / "gui"
    output_root = tmp_path / "outputs" / "gui" / "runs"
    monkeypatch.setenv("VBOGS_GUI_DATA_ROOT", str(data_root))
    monkeypatch.setenv("VBOGS_GUI_OUTPUT_ROOT", str(output_root))
    return web_app.create_app(root=REPO_ROOT, store_path=tmp_path / "control.sqlite3")


def make_run(app, run_id: str = "run-delete-test") -> dict:
    workspace = app.state.data_root / "runs" / run_id
    output = app.state.output_root / run_id
    workspace.mkdir(parents=True)
    output.mkdir(parents=True)
    (workspace / "pipeline.log").write_text("run output\n", encoding="utf-8")
    (output / "scene").mkdir()
    (output / "scene" / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    return {
        "id": run_id,
        "owner": "operator@example.test",
        "dataset": "kitti360",
        "scene_id": "scene",
        "preset": "kitti360-dev",
        "start_at": "prepare",
        "stop_after": "bundle",
        "created_at": "2026-01-01T00:00:00+00:00",
        "config_path": str(workspace / "resolved_config.yaml"),
        "workspace_path": str(workspace),
        "output_path": str(output),
        "command": ["scripts/run_pipeline.sh"],
    }


@pytest.mark.parametrize("status", ["queued", "cancelled", "completed", "failed", "interrupted"])
def test_delete_eligible_run_removes_its_files_and_record(tmp_path, monkeypatch, status):
    from fastapi.testclient import TestClient

    app = make_app(tmp_path, monkeypatch)
    run = make_run(app, f"run-delete-{status}")
    app.state.store.create_run(run)
    if status != "queued":
        app.state.store.transition(run["id"], status)

    response = TestClient(app).delete(f"/api/runs/{run['id']}", headers=HEADERS, json={"confirm_run_id": run["id"]})

    assert response.status_code == 200
    assert response.json() == {"id": run["id"], "deleted": True}
    assert not Path(run["workspace_path"]).exists()
    assert not Path(run["output_path"]).exists()
    assert app.state.store.get_run(run["id"]) is None
    assert app.state.store.events(run["id"]) == []


def test_delete_rejects_wrong_confirmation_and_other_operators(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = make_app(tmp_path, monkeypatch)
    run = make_run(app)
    app.state.store.create_run(run)
    client = TestClient(app)

    wrong_confirmation = client.delete(f"/api/runs/{run['id']}", headers=HEADERS, json={"confirm_run_id": "wrong"})
    unauthorised = client.delete(f"/api/runs/{run['id']}", headers={"X-Forwarded-User": "other@example.test"}, json={"confirm_run_id": run["id"]})

    assert wrong_confirmation.status_code == 422
    assert unauthorised.status_code == 403
    assert Path(run["workspace_path"]).is_dir()
    assert Path(run["output_path"]).is_dir()
    assert app.state.store.get_run(run["id"]) is not None


@pytest.mark.parametrize("status", ["starting", "running", "cancelling"])
def test_delete_rejects_active_runs_without_removing_files(tmp_path, monkeypatch, status):
    from fastapi.testclient import TestClient

    app = make_app(tmp_path, monkeypatch)
    run = make_run(app, f"run-active-{status}")
    app.state.store.create_run(run)
    app.state.store.transition(run["id"], status)

    response = TestClient(app).delete(f"/api/runs/{run['id']}", headers=HEADERS, json={"confirm_run_id": run["id"]})

    assert response.status_code == 409
    assert Path(run["workspace_path"]).is_dir()
    assert Path(run["output_path"]).is_dir()
    assert app.state.store.get_run(run["id"]) is not None


def test_delete_rejects_storage_path_outside_gui_roots(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = make_app(tmp_path, monkeypatch)
    run = make_run(app)
    outside = tmp_path / "outside-workspace"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    run["workspace_path"] = str(outside)
    app.state.store.create_run(run)

    response = TestClient(app).delete(f"/api/runs/{run['id']}", headers=HEADERS, json={"confirm_run_id": run["id"]})

    assert response.status_code == 422
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert app.state.store.get_run(run["id"]) is not None


def test_delete_stops_active_viewer_before_removing_completed_run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = make_app(tmp_path, monkeypatch)
    run = make_run(app)
    app.state.store.create_run(run)
    app.state.store.transition(run["id"], "completed")
    app.state.store.set_viewer(run["id"], "0", "operator@example.test")
    stopped: list[bool] = []
    monkeypatch.setattr(web_app, "_stop_shared_viewer", lambda: stopped.append(True))

    response = TestClient(app).delete(f"/api/runs/{run['id']}", headers=HEADERS, json={"confirm_run_id": run["id"]})

    assert response.status_code == 200
    assert stopped == [True]
    assert app.state.store.viewer()["status"] == "idle"
    assert app.state.store.get_run(run["id"]) is None
