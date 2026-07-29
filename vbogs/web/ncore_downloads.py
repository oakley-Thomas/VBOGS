"""Serialized NCore downloads with browser-supplied, volatile credentials."""

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
    """Run one full-clip transfer at a time without persisting credentials."""

    def __init__(
        self,
        store: RunStore,
        *,
        ncore_root: Path = DEFAULT_NCORE_ROOT,
        catalog_loader: Callable[[str], list[str]] | None = None,
        downloader: Callable[..., bool] = download_full_scene,
    ):
        self.store = store
        self.ncore_root = ncore_root
        self.catalog_loader = catalog_loader or self._load_catalog
        self.downloader = downloader
        self.task: asyncio.Task[None] | None = None
        self.loop_task: asyncio.Task[None] | None = None
        self.wake = asyncio.Event()
        # Deliberately in-memory only. A restart interrupts queued work because
        # no submitted credential is recovered from disk or configuration.
        self._credentials: dict[str, str] = {}

    def start(self) -> None:
        self.store.mark_active_downloads_interrupted()
        self.loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.loop_task:
            self.loop_task.cancel()
        if self.task:
            self.task.cancel()
        await asyncio.gather(*(task for task in (self.loop_task, self.task) if task), return_exceptions=True)

    @staticmethod
    def _require_token(token: str | None) -> str:
        value = (token or "").strip()
        if not value:
            raise ValueError("Paste a Hugging Face read token to search or download NCore clips.")
        return value

    def _load_catalog(self, token: str) -> list[str]:
        return discover_scene_ids(list_repo_files(DEFAULT_REPO_ID, "main", token))

    async def catalog(self, *, token: str | None, query: str = "", limit: int = 100) -> list[str]:
        """List only clips authorized by this request's submitted token."""

        scenes = sorted(set(await asyncio.to_thread(self.catalog_loader, self._require_token(token))))
        needle = query.strip().lower()
        selected = (scene for scene in scenes if not needle or needle in scene.lower())
        return list(selected)[:max(1, min(limit, 500))]

    def enqueue(self, *, scene_id: str, owner: str, token: str | None) -> dict:
        credential = self._require_token(token)
        if not NCORE_SCENE_RE.fullmatch(scene_id):
            raise ValueError("NCore clip ID must be a UUID from the authorized catalog.")
        if self.store.active_download_for_scene(scene_id):
            raise ValueError("This clip already has a queued or running download.")
        record = self.store.create_download({
            "id": f"ncore-{uuid.uuid4().hex[:12]}", "owner": owner, "scene_id": scene_id, "created_at": utc_now(),
        })
        self._credentials[record["id"]] = credential
        self.wake.set()
        return record

    def _safe(self, message: object, *credentials: str | None) -> str:
        text = str(message).replace("\x00", " ")
        for credential in (*self._credentials.values(), *credentials):
            if credential:
                text = text.replace(credential, "[redacted]")
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
        token: str | None = None
        try:
            token = self._credentials.get(download["id"])
            token = self._require_token(token)
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
            error = self._safe(exc, token)
            self.store.transition_download(download["id"], "failed", error=error)
            self.store.add_download_event(download["id"], f"Download failed: {error}")
        finally:
            self._credentials.pop(download["id"], None)
            self.wake.set()
