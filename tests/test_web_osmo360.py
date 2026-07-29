from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

import vbogs.web.app as web_app


REPO_ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"X-Forwarded-User": "operator@example.test"}


def make_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VBOGS_GUI_DATA_ROOT", str(tmp_path / "data" / "gui"))
    monkeypatch.setenv("VBOGS_GUI_OUTPUT_ROOT", str(tmp_path / "outputs" / "gui" / "runs"))
    return web_app.create_app(root=REPO_ROOT, store_path=tmp_path / "control.sqlite3")


def test_osmo_upload_creates_only_a_run_owned_staged_input(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = make_app(tmp_path, monkeypatch)
    response = TestClient(app).post(
        "/api/osmo360/runs", headers=HEADERS,
        data={"scene_id": "living-room", "profile": "balanced"},
        files={"video": ("capture.mp4", b"not-yet-probed", "video/mp4")},
    )

    assert response.status_code == 201
    run = response.json()
    assert run["workflow"] == "osmo360_splat"
    assert run["dataset"] == "osmo360"
    workspace = Path(run["workspace_path"])
    assert (workspace / "input" / "source.mp4").read_bytes() == b"not-yet-probed"
    assert (workspace / "upload.json").is_file()
    assert not Path(run["output_path"]).exists()


@pytest.mark.parametrize("scene_id,filename", [("../bad", "capture.mp4"), ("valid", "capture.avi")])
def test_osmo_upload_rejects_unsafe_scene_or_unsupported_video(tmp_path, monkeypatch, scene_id, filename):
    from fastapi.testclient import TestClient

    app = make_app(tmp_path, monkeypatch)
    response = TestClient(app).post(
        "/api/osmo360/runs", headers=HEADERS,
        data={"scene_id": scene_id, "profile": "balanced"},
        files={"video": (filename, b"payload", "video/mp4")},
    )

    assert response.status_code == 422
    assert app.state.store.list_runs() == []
