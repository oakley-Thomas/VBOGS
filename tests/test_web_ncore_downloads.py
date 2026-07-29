from pathlib import Path
import time

import pytest

pytest.importorskip("fastapi")

import vbogs.web.app as web_app


SCENE_ID = "00b769dd-b4fa-4d88-ba4e-e6a230ff0c66"
HEADERS = {"X-Forwarded-User": "operator@example.test"}


def test_ncore_api_is_operator_only_and_never_returns_backend_token(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("VBOGS_GUI_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("VBOGS_GUI_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("VBOGS_NCORE_HF_TOKEN", "hf-private-token")
    monkeypatch.setenv("VBOGS_GUI_ADMINS", "operator@example.test")
    monkeypatch.setenv("VBOGS_GUI_VIEWERS", "viewer@example.test")
    app = web_app.create_app(root=Path(__file__).resolve().parents[1], store_path=tmp_path / "control.sqlite3")
    app.state.ncore_downloads.ncore_root = tmp_path / "ncore"
    app.state.ncore_downloads.catalog_loader = lambda token: [SCENE_ID]

    def fake_download(scene_id, **kwargs):
        root = kwargs["ncore_root"] / scene_id
        root.mkdir(parents=True)
        for suffix in ("json", "ncore4.zarr.itar", "ncore4-camera_front_wide_120fov.zarr.itar", "ncore4-camera_front_tele_30fov.zarr.itar", "ncore4-lidar_top_360fov.zarr.itar"):
            (root / f"pai_{scene_id}.{suffix}").write_text("fixture", encoding="utf-8")
        kwargs["progress"]("token hf-private-token is redacted")
        return True

    app.state.ncore_downloads.downloader = fake_download
    with TestClient(app) as client:
        assert client.get("/api/ncore/catalog", headers={"X-Forwarded-User": "viewer@example.test"}).status_code == 403
        catalog = client.get("/api/ncore/catalog", headers=HEADERS)
        assert catalog.json()["clips"] == [{"scene_id": SCENE_ID, "status": "missing"}]
        created = client.post("/api/ncore/downloads", headers=HEADERS, json={"scene_id": SCENE_ID})
        assert created.status_code == 201
        download_id = created.json()["id"]
        for _ in range(50):
            jobs = client.get("/api/ncore/downloads", headers=HEADERS).json()
            if jobs[0]["status"] == "completed":
                break
            time.sleep(0.02)
        assert jobs[0]["status"] == "completed"
        events = client.get(f"/api/ncore/downloads/{download_id}/log", headers=HEADERS).json()["events"]
        assert "hf-private-token" not in str(events)
        assert client.post("/api/ncore/downloads", headers=HEADERS, json={"scene_id": "22b769dd-b4fa-4d88-ba4e-e6a230ff0c66"}).status_code == 422
