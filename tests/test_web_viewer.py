import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import vbogs.web.app as web_app
from vbogs.web.app import WebError, resolve_viewer_inputs, viewer_readiness
from vbogs.web.store import RunStore


def run_record(tmp_path: Path, *, status: str = "completed") -> dict:
    workspace = tmp_path / "workspace"
    return {
        "id": "run-viewer-test",
        "owner": "operator@example.test",
        "status": status,
        "scene_id": "scene",
        "workspace_path": str(workspace),
        "output_path": str(tmp_path / "output"),
    }


def write_workspace_artifacts(run: dict) -> tuple[Path, Path]:
    artifacts = Path(run["workspace_path"]) / "artifacts"
    model = artifacts / "octree" / "scene" / "train-a"
    model.mkdir(parents=True)
    (model / "config.yaml").write_text("model: test\n", encoding="utf-8")
    uncertainty = artifacts / "m4" / run["scene_id"] / "U.npy"
    uncertainty.parent.mkdir(parents=True)
    uncertainty.write_bytes(b"numpy-placeholder")
    (artifacts / "train_run.json").write_text(json.dumps({"model_path": str(model)}), encoding="utf-8")
    return model, uncertainty


def test_workspace_artifacts_make_completed_run_viewer_ready(tmp_path):
    run = run_record(tmp_path)
    model, uncertainty = write_workspace_artifacts(run)

    resolved = resolve_viewer_inputs(run)

    assert resolved.model_path == model.resolve()
    assert resolved.uncertainty_path == uncertainty.resolve()
    assert resolved.source == "run_workspace"
    assert viewer_readiness(run) == {"ready": True, "reason": None, "source": "run_workspace"}


def test_portable_export_is_preferred_over_workspace_artifacts(tmp_path):
    run = run_record(tmp_path)
    write_workspace_artifacts(run)
    portable = Path(run["output_path"]) / run["scene_id"] / "local_viewer"
    (portable / "model").mkdir(parents=True)
    (portable / "model" / "config.yaml").write_text("model: export\n", encoding="utf-8")
    (portable / "uncertainty").mkdir()
    (portable / "uncertainty" / "U.npy").write_bytes(b"export")

    assert resolve_viewer_inputs(run).source == "portable_export"


def test_viewer_rejects_training_record_outside_run_workspace(tmp_path):
    run = run_record(tmp_path)
    _, uncertainty = write_workspace_artifacts(run)
    outside = tmp_path / "outside-model"
    outside.mkdir()
    (outside / "config.yaml").write_text("model: outside\n", encoding="utf-8")
    record = Path(run["workspace_path"]) / "artifacts" / "train_run.json"
    record.write_text(json.dumps({"model_path": str(outside)}), encoding="utf-8")

    with pytest.raises(WebError, match="outside this run"):
        resolve_viewer_inputs(run)
    assert uncertainty.is_file()


def test_viewer_state_tracks_revision_and_releases_gpu(tmp_path):
    store = RunStore(tmp_path / "control.sqlite3")

    active = store.set_viewer("run-a", "1", "operator")
    idle = store.clear_viewer()

    assert active["status"] == "active"
    assert active["revision"] == 1
    assert idle["status"] == "idle"
    assert idle["run_id"] is None
    assert idle["revision"] == 2


def test_console_viewer_loads_ready_run_and_proxies_only_for_authenticated_user(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    run = run_record(tmp_path)
    write_workspace_artifacts(run)
    run.update({
        "dataset": "kitti360", "preset": "kitti360-dev", "start_at": "prepare", "stop_after": "uncertainty",
        "created_at": "2026-01-01T00:00:00+00:00", "config_path": str(tmp_path / "config.yaml"),
        "command": ["scripts/run_pipeline.sh"],
    })
    started: list[tuple[str, str]] = []
    monkeypatch.setattr(web_app, "_start_shared_viewer", lambda record, gpu, inputs: started.append((record["id"], gpu)))
    monkeypatch.setattr(web_app, "_stop_shared_viewer", lambda: None)

    app = web_app.create_app(root=Path(__file__).resolve().parents[1], store_path=tmp_path / "control.sqlite3")
    app.state.store.create_run(run)
    app.state.store.transition(run["id"], "completed")

    async def fake_proxy(method, path, payload=None):
        return {"path": path, "method": method}

    monkeypatch.setattr(web_app, "proxy_renderer_json", fake_proxy)
    headers = {"X-Forwarded-User": "operator@example.test"}
    with TestClient(app) as client:
        readiness = client.get(f"/api/runs/{run['id']}/viewer-readiness", headers=headers)
        assert readiness.json()["ready"] is True
        assert client.get("/api/viewer/metadata").status_code == 401

        loaded = client.post("/api/viewer", headers=headers, json={"run_id": run["id"]})
        assert loaded.status_code == 200
        assert loaded.json()["gpu_id"] == "0"
        assert started == [(run["id"], "0")]

        metadata = client.get("/api/viewer/metadata", headers=headers)
        assert metadata.json() == {"path": "/api/metadata", "method": "GET"}
        stopped = client.delete("/api/viewer", headers=headers)
        assert stopped.json()["status"] == "idle"


def test_runs_api_separates_active_queue_from_completed_catalog(tmp_path):
    from fastapi.testclient import TestClient

    app = web_app.create_app(root=Path(__file__).resolve().parents[1], store_path=tmp_path / "control.sqlite3")
    active = {
        **run_record(tmp_path, status="queued"),
        "dataset": "kitti360", "preset": "kitti360-dev", "start_at": "prepare", "stop_after": "bundle",
        "created_at": "2026-01-01T00:00:00+00:00", "config_path": str(tmp_path / "config.yaml"),
        "command": ["scripts/run_pipeline.sh"],
    }
    completed = {**active, "id": "run-completed", "workspace_path": str(tmp_path / "completed-workspace")}
    app.state.store.create_run(active)
    app.state.store.create_run(completed)
    app.state.store.transition(active["id"], "starting", gpu_id="0")
    app.state.store.transition(completed["id"], "completed")
    headers = {"X-Forwarded-User": "operator@example.test"}

    with TestClient(app) as client:
        assert [run["id"] for run in client.get("/api/runs?scope=active", headers=headers).json()] == [active["id"]]
        assert [run["id"] for run in client.get("/api/runs?scope=completed", headers=headers).json()] == [completed["id"]]
        assert client.get("/api/runs?scope=unknown", headers=headers).status_code == 422


def test_run_detail_exposes_safe_progress_and_streams_progress_event(tmp_path):
    from fastapi.testclient import TestClient

    app = web_app.create_app(root=Path(__file__).resolve().parents[1], store_path=tmp_path / "control.sqlite3")
    run = {
        **run_record(tmp_path), "dataset": "kitti360", "preset": "kitti360-dev", "start_at": "prepare", "stop_after": "train",
        "created_at": "2026-01-01T00:00:00+00:00", "config_path": str(tmp_path / "config.yaml"),
        "command": ["scripts/run_pipeline.sh"],
    }
    Path(run["workspace_path"]).mkdir()
    (Path(run["workspace_path"]) / "pipeline.events.jsonl").write_text(
        '{"type":"run_started","stages":["prepare","train"]}\n', encoding="utf-8"
    )
    app.state.store.create_run(run)
    app.state.store.transition(run["id"], "completed")
    headers = {"X-Forwarded-User": "operator@example.test"}

    with TestClient(app) as client:
        detail = client.get(f"/api/runs/{run['id']}", headers=headers)
        assert detail.status_code == 200
        progress = detail.json()["progress"]
        assert progress["overall"]["percent"] == 100.0
        assert "workspace_path" not in progress

        stream = client.get(f"/api/runs/{run['id']}/events/stream", headers=headers)
        assert stream.status_code == 200
        assert "event: progress" in stream.text


def test_resume_clears_stale_training_progress_snapshot(tmp_path):
    from fastapi.testclient import TestClient

    app = web_app.create_app(root=Path(__file__).resolve().parents[1], store_path=tmp_path / "control.sqlite3")
    run = {
        **run_record(tmp_path), "dataset": "kitti360", "preset": "kitti360-dev", "start_at": "train", "stop_after": "train",
        "created_at": "2026-01-01T00:00:00+00:00", "config_path": str(tmp_path / "config.yaml"),
        "command": ["scripts/run_pipeline.sh"],
    }
    workspace = Path(run["workspace_path"])
    (workspace / "artifacts" / "colmap" / run["scene_id"]).mkdir(parents=True)
    (workspace / "artifacts" / "colmap" / run["scene_id"] / "metadata.json").write_text("{}", encoding="utf-8")
    stale_snapshot = workspace / "training_progress.json"
    stale_snapshot.write_text('{"state":"running"}', encoding="utf-8")
    app.state.store.create_run(run)
    app.state.store.transition(run["id"], "failed")

    with TestClient(app) as client:
        response = client.post(
            f"/api/runs/{run['id']}/resume", headers={"X-Forwarded-User": "operator@example.test"},
            json={"start_at": "train", "stop_after": "train"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert not stale_snapshot.exists()
