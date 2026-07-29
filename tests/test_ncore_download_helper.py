import asyncio
from pathlib import Path

import pytest

import vbogs.ncore_download as ncore_download
from vbogs.ncore_download import discover_scene_ids, download_scene, files_for_scene
from vbogs.web.ncore_downloads import NCoreDownloadManager
from vbogs.web.store import RunStore, utc_now


SCENE_ID = "00b769dd-b4fa-4d88-ba4e-e6a230ff0c66"
REPO_ROOT = Path(__file__).resolve().parents[1]


def remote_files(scene_id: str = SCENE_ID) -> list[str]:
    prefix = f"clips/{scene_id}/"
    return [
        prefix + f"pai_{scene_id}.json",
        prefix + f"pai_{scene_id}.ncore4.zarr.itar",
        prefix + f"pai_{scene_id}.ncore4-camera_front_wide_120fov.zarr.itar",
        prefix + f"pai_{scene_id}.ncore4-camera_front_tele_30fov.zarr.itar",
        prefix + f"pai_{scene_id}.ncore4-lidar_top_360fov.zarr.itar",
    ]


def test_catalog_discovers_only_real_ncore_scene_components():
    paths = remote_files() + ["README.md", f"clips/{SCENE_ID}/other.bin"]
    assert discover_scene_ids(paths) == [SCENE_ID]
    assert files_for_scene(SCENE_ID, paths, "full") == sorted(path for path in paths if path.startswith(f"clips/{SCENE_ID}/"))


def test_full_download_retains_existing_components_and_completes_missing(tmp_path, monkeypatch):
    root = tmp_path / "ncore"
    existing = root / SCENE_ID / f"pai_{SCENE_ID}.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing", encoding="utf-8")
    calls: list[str] = []

    def fake_download(_repo, _revision, _token, remote_path, destination, _progress=None):
        calls.append(remote_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(remote_path, encoding="utf-8")
        return destination

    monkeypatch.setattr(ncore_download, "download_remote_file", fake_download)
    assert download_scene(SCENE_ID, remote_files(), token="secret", ncore_root=root)
    assert existing.read_text(encoding="utf-8") == "existing"
    assert len(calls) == 4
    assert all((root / SCENE_ID / Path(path).name).is_file() for path in remote_files())


def test_download_manager_redacts_token_and_marks_restart_interrupted(tmp_path):
    store = RunStore(tmp_path / "control.sqlite3")
    record = store.create_download({"id": "ncore-running", "owner": "operator", "scene_id": SCENE_ID, "created_at": utc_now()})
    store.transition_download(record["id"], "running")
    assert store.mark_active_downloads_interrupted() == 1
    assert store.get_download(record["id"])["status"] == "interrupted"

    manager = NCoreDownloadManager(store, ncore_root=tmp_path)
    manager.enqueue(scene_id=SCENE_ID, owner="operator", token="hf-super-secret")
    manager._progress(record["id"], "Bearer hf-super-secret was rejected")
    assert "hf-super-secret" not in store.download_events(record["id"])[-1]["message"]


def test_download_manager_serializes_queued_downloads(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "control.sqlite3")
    completed: list[str] = []

    def fake_download(scene_id, **kwargs):
        kwargs["progress"](f"downloading {scene_id}")
        completed.append(scene_id)
        return True

    manager = NCoreDownloadManager(store, ncore_root=tmp_path, downloader=fake_download)

    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)

    async def exercise():
        manager.start()
        manager.enqueue(scene_id=SCENE_ID, owner="operator", token="hf-token")
        other = "11b769dd-b4fa-4d88-ba4e-e6a230ff0c66"
        manager.enqueue(scene_id=other, owner="operator", token="hf-token")
        for _ in range(50):
            if len(completed) == 2:
                break
            await asyncio.sleep(0.02)
        await manager.stop()

    asyncio.run(exercise())
    assert completed == [SCENE_ID, "11b769dd-b4fa-4d88-ba4e-e6a230ff0c66"]
    assert [job["status"] for job in store.list_downloads()] == ["completed", "completed"]


def test_download_manager_rejects_duplicate_active_scene(tmp_path):
    store = RunStore(tmp_path / "control.sqlite3")
    manager = NCoreDownloadManager(store, ncore_root=tmp_path)
    manager.enqueue(scene_id=SCENE_ID, owner="operator", token="hf-token")
    with pytest.raises(ValueError, match="already"):
        manager.enqueue(scene_id=SCENE_ID, owner="operator", token="hf-token")


def test_compose_does_not_configure_an_ncore_token():
    for compose_name in ("compose.yml", "deploy.yml"):
        text = (REPO_ROOT / "docker" / "compose" / compose_name).read_text(encoding="utf-8")
        assert "VBOGS_NCORE_HF_TOKEN" not in text
    assert "VBOGS_NCORE_HF_TOKEN=" not in (REPO_ROOT / "configs" / "docker" / "stack.env").read_text(encoding="utf-8")
