"""FIFO, one-run-per-GPU scheduler for validated pipeline commands."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Awaitable, Callable

from vbogs.web.store import RunStore


Runner = Callable[[dict, str], Awaitable[int]]


class Scheduler:
    def __init__(self, store: RunStore, gpu_ids: tuple[str, ...], runner: Runner):
        self.store = store
        self.gpu_ids = gpu_ids
        self.runner = runner
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.wake = asyncio.Event()
        self.loop_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.store.mark_active_interrupted()
        self.loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.loop_task:
            self.loop_task.cancel()
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*(self.tasks.values()), return_exceptions=True)

    def notify(self) -> None:
        self.wake.set()

    def slots(self) -> list[dict[str, str | None]]:
        active = {str(run["gpu_id"]): run["id"] for run in self.store.active_runs() if run["gpu_id"] is not None}
        viewer = self.store.viewer()
        return [
            {"gpu_id": gpu, "run_id": active.get(gpu), "viewer_run_id": viewer.get("run_id") if viewer and viewer.get("gpu_id") == gpu else None}
            for gpu in self.gpu_ids
        ]

    async def _loop(self) -> None:
        while True:
            self._dispatch()
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    def _dispatch(self) -> None:
        occupied = {str(run["gpu_id"]) for run in self.store.active_runs() if run["gpu_id"] is not None}
        viewer = self.store.viewer()
        if viewer and viewer.get("gpu_id"):
            occupied.add(str(viewer["gpu_id"]))
        available = [gpu for gpu in self.gpu_ids if gpu not in occupied]
        for run, gpu in zip(self.store.queued_runs(), available):
            self.store.transition(run["id"], "starting", gpu_id=gpu)
            self.store.add_event(run["id"], "assigned", {"gpu_id": gpu})
            task = asyncio.create_task(self._run(run["id"], gpu))
            self.tasks[run["id"]] = task

    async def _run(self, run_id: str, gpu_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            return
        self.store.transition(run_id, "running", gpu_id=gpu_id)
        self.store.add_event(run_id, "started", {"gpu_id": gpu_id})
        try:
            code = await self.runner(run, gpu_id)
            latest = self.store.get_run(run_id)
            if latest and latest["cancel_requested"]:
                self.store.transition(run_id, "cancelled")
                self.store.add_event(run_id, "cancelled", {})
            elif code == 0:
                self.store.transition(run_id, "completed")
                self.store.add_event(run_id, "completed", {})
            else:
                self.store.transition(run_id, "failed", error=f"Pipeline exited with status {code}")
                self.store.add_event(run_id, "failed", {"exit_code": code})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.store.transition(run_id, "failed", error=str(exc))
            self.store.add_event(run_id, "failed", {"error": str(exc)})
        finally:
            self.tasks.pop(run_id, None)
            self.notify()

    async def cancel(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        self.store.request_cancel(run_id)
        if run["status"] == "queued":
            self.store.transition(run_id, "cancelled")
            self.store.add_event(run_id, "cancelled", {"before_start": True})
            return
        if run["status"] in {"starting", "running"}:
            self.store.transition(run_id, "cancelling")
            Path(run["workspace_path"], "cancel.request").touch()
            self.store.add_event(run_id, "cancelling", {})
        self.notify()


async def subprocess_runner(run: dict, gpu_id: str) -> int:
    """Run the existing pipeline entrypoint and retain an unfiltered job log."""
    workspace = Path(run["workspace_path"])
    log_path = workspace / "pipeline.log"
    event_path = workspace / "pipeline.events.jsonl"
    progress_path = workspace / "training_progress.json"
    command = [
        "scripts/run_pipeline.sh", "--config", run["config_path"],
        "--gpu", gpu_id, "--jax-device", gpu_id,
        "--artifact-root", str(workspace / "artifacts"),
        "--run-output-root", run["output_path"],
        "--start-at", run["start_at"], "--stop-after", run["stop_after"],
        "--event-log", str(event_path), "--progress-path", str(progress_path),
        "--cancel-file", str(workspace / "cancel.request"),
    ]
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=handle, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "VBOGS_GUI_RUN_ID": run["id"]},
        )
        while True:
            try:
                return await asyncio.wait_for(process.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                # The pipeline runner polls this marker and sends TERM/KILL to
                # its recorded in-container stage process group. Do not signal
                # the wrapper here: doing so could orphan a docker-exec child.
                continue
