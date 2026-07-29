"""Serialized, persisted NCore downloads for the authenticated web console."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Callable

from vbogs.ncore_download import DEFAULT_NCORE_ROOT, DEFAULT_REPO_ID, discover_scene_ids, download_full_scene, list_repo_files
from vbogs.web.store import RunStore, utc_now


NCORE_SCENE_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class NCoreDownloadManager:
    """Run one full-clip transfer at a time without exposing its credential."""

    def __init__(
        self,
        store: RunStore,
        *,
        token: str | None,
        ncore_root: Path = DEFAULT_NCORE_ROOT,
        catalog_ttl_seconds: float = 900.0,
        catalog_loader: Callable[[str], list[str]] | None = None,
        downloader: Callable[..., bool] = download_full_scene,
    ):
        self.store = store
        self.token = token
        self.ncore_root = ncore_root
        self.catalog_ttl_seconds = catalog_ttl_seconds
        self.catalog_loader = catalog_loader or self._load_catalog
        self.downloader = downloader
        self.task: asyncio.Task[None] | None = None
        self.loop_task: asyncio.Task[None] | None = None
        self.wake = asyncio.Event()
        self.catalog_cache: tuple[float, list[str]] | None = None

    def start(self) -> None:
        self.store.mark_active_downloads_interrupted()
        self.loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.loop_task:
            self.loop_task.cancel()
        if self.task:
            self.task.cancel()
        await asyncio.gather(*(task for task in (self.loop_task, self.task) if task), return_exceptions=True)

    def _require_token(self) -> str:
        if not self.token:
            raise ValueError("NCore downloads are not configured. Set VBOGS_NCORE_HF_TOKEN on the web service.")
        return self.token

    def _load_catalog(self, token: str) -> list[str]:
        return discover_scene_ids(list_repo_files(DEFAULT_REPO_ID, "main", token))

    async def catalog(self, *, query: str = "", limit: int = 100) -> list[str]:
        token = self._require_token()
        now = asyncio.get_running_loop().time()
        if self.catalog_cache is None or now - self.catalog_cache[0] >= self.catalog_ttl_seconds:
            scenes = await asyncio.to_thread(self.catalog_loader, token)
            self.catalog_cache = (now, sorted(set(scenes)))
        needle = query.strip().lower()
        selected = (scene for scene in self.catalog_cache[1] if not needle or needle in scene.lower())
        return list(selected)[:max(1, min(limit, 500))]

    def enqueue(self, *, scene_id: str, owner: str) -> dict:
        self._require_token()
        if not NCORE_SCENE_RE.fullmatch(scene_id):
            raise ValueError("NCore clip ID must be a UUID from the authorized catalog.")
        if self.store.active_download_for_scene(scene_id):
            raise ValueError("This clip already has a queued or running download.")
        record = self.store.create_download({
            "id": f"ncore-{uuid.uuid4().hex[:12]}", "owner": owner, "scene_id": scene_id, "created_at": utc_now(),
        })
        self.wake.set()
        return record

    def _safe(self, message: object) -> str:
        text = str(message).replace("\x00", " ")
        if self.token:
            text = text.replace(self.token, "[redacted]")
        return text[:4000]

    def _progress(self, download_id: str, message: str) -> None:
        self.store.add_download_event(download_id, self._safe(message))

    async def _loop(self) -> None:
        while True:
            if self.task is None or self.task.done():
                queued = self.store.next_queued_download()
                if queued:
                    self.store.transition_download(queued["id"], "running")
                    self.store.add_download_event(queued["id"], "Download started.")
                    self.task = asyncio.create_task(self._run(queued))
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    async def _run(self, download: dict) -> None:
        try:
            token = self._require_token()
            completed = await asyncio.to_thread(
                self.downloader, download["scene_id"], token=token, ncore_root=self.ncore_root,
                progress=lambda message: self._progress(download["id"], message),
            )
            if not completed:
                raise RuntimeError("The NCore downloader did not complete the requested clip.")
            self.store.transition_download(download["id"], "completed")
            self.store.add_download_event(download["id"], "Download completed.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = self._safe(exc)
            self.store.transition_download(download["id"], "failed", error=error)
            self.store.add_download_event(download["id"], f"Download failed: {error}")
        finally:
            self.wake.set()
