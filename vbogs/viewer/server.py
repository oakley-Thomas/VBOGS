"""FastAPI app for the realtime Octree-AnyGS debug viewer."""

import asyncio
from pathlib import Path
from typing import Any

from vbogs.viewer.rendering import OctreeRenderSession

STATIC_DIR = Path(__file__).resolve().parent / "static"


class LatestRequestBuffer:
    """One-slot request buffer used to drop stale freefly camera updates."""

    def __init__(self) -> None:
        self._latest: dict[str, Any] | None = None

    def replace(self, request: dict[str, Any]) -> None:
        self._latest = request

    def take_latest(self) -> dict[str, Any] | None:
        request = self._latest
        self._latest = None
        return request

    def is_empty(self) -> bool:
        return self._latest is None


def create_app(session: OctreeRenderSession):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="VBOGS Realtime Viewer")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/metadata")
    async def metadata():
        return JSONResponse(session.metadata())

    @app.get("/api/cameras")
    async def cameras():
        return JSONResponse(session.camera_payload())

    @app.websocket("/ws/render")
    async def render_socket(websocket: WebSocket):
        await websocket.accept()
        latest = LatestRequestBuffer()
        event = asyncio.Event()
        closed = asyncio.Event()

        async def receive_loop() -> None:
            try:
                while True:
                    payload = await websocket.receive_json()
                    if not isinstance(payload, dict):
                        payload = {"request_id": None, "layer": "rgb"}
                    latest.replace(payload)
                    event.set()
            except WebSocketDisconnect:
                closed.set()
            except Exception:
                closed.set()

        receiver = asyncio.create_task(receive_loop())
        try:
            while not closed.is_set():
                wait_closed = asyncio.create_task(closed.wait())
                wait_request = asyncio.create_task(event.wait())
                done, pending = await asyncio.wait(
                    {wait_closed, wait_request},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if wait_closed in done:
                    break

                request = latest.take_latest()
                if latest.is_empty():
                    event.clear()
                if request is None:
                    continue

                try:
                    frame = await asyncio.to_thread(session.render_request, request)
                    await websocket.send_json(frame.metadata)
                    await websocket.send_bytes(frame.jpeg)
                except WebSocketDisconnect:
                    closed.set()
                except Exception as exc:
                    await websocket.send_json(
                        {
                            "request_id": request.get("request_id"),
                            "error": str(exc),
                        }
                    )
        finally:
            receiver.cancel()

    return app
