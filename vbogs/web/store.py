"""SQLite persistence for the single-instance VBOGS web scheduler."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    preset TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    stop_after TEXT NOT NULL,
                    gpu_id TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    config_path TEXT NOT NULL,
                    workspace_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                CREATE TABLE IF NOT EXISTS viewer_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    run_id TEXT,
                    gpu_id TEXT,
                    owner TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _run(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        value["command"] = json.loads(value.pop("command_json"))
        value["cancel_requested"] = bool(value["cancel_requested"])
        return value

    def create_run(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO runs (
                    id, owner, status, dataset, scene_id, preset, start_at, stop_after,
                    gpu_id, created_at, config_path, workspace_path, output_path, command_json
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)""",
                (
                    record["id"], record["owner"], record["dataset"], record["scene_id"],
                    record["preset"], record["start_at"], record["stop_after"], record["created_at"],
                    record["config_path"], record["workspace_path"], record["output_path"],
                    json.dumps(record["command"]),
                ),
            )
        self.add_event(record["id"], "queued", {"message": "Run queued"})
        return self.get_run(record["id"]) or record

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._run(connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._run(row) for row in rows if row is not None]

    def queued_runs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs WHERE status = 'queued' ORDER BY created_at ASC").fetchall()
        return [self._run(row) for row in rows if row is not None]

    def active_runs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs WHERE status IN ('starting', 'running', 'cancelling')").fetchall()
        return [self._run(row) for row in rows if row is not None]

    def transition(self, run_id: str, status: str, *, gpu_id: str | None = None, error: str | None = None) -> None:
        now = utc_now()
        fields = ["status = ?"]
        values: list[Any] = [status]
        if gpu_id is not None:
            fields.append("gpu_id = ?")
            values.append(gpu_id)
        if status == "running":
            fields.append("started_at = COALESCE(started_at, ?)")
            values.append(now)
        if status in TERMINAL_STATUSES:
            fields.append("finished_at = ?")
            values.append(now)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        values.append(run_id)
        with self._transaction() as connection:
            connection.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id = ?", values)

    def request_cancel(self, run_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("UPDATE runs SET cancel_requested = 1 WHERE id = ?", (run_id,))

    def requeue(self, run_id: str, *, start_at: str, stop_after: str) -> dict[str, Any] | None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE runs SET status = 'queued', start_at = ?, stop_after = ?, gpu_id = NULL,
                   started_at = NULL, finished_at = NULL, error = NULL, cancel_requested = 0
                   WHERE id = ?""",
                (start_at, stop_after, run_id),
            )
        self.add_event(run_id, "requeued", {"start_at": start_at, "stop_after": stop_after})
        return self.get_run(run_id)

    def add_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO run_events(run_id, created_at, type, payload_json) VALUES (?, ?, ?, ?)",
                (run_id, utc_now(), event_type, json.dumps(payload, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, created_at, type, payload_json FROM run_events WHERE run_id = ? AND sequence > ? ORDER BY sequence",
                (run_id, after),
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def mark_active_interrupted(self) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status = 'interrupted', finished_at = ?, error = ? WHERE status IN ('starting', 'running', 'cancelling')",
                (utc_now(), "Web scheduler restarted; resume this run from a completed stage."),
            )
            return cursor.rowcount

    def set_viewer(self, run_id: str | None, gpu_id: str | None, owner: str | None) -> None:
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO viewer_state(singleton, run_id, gpu_id, owner, updated_at)
                   VALUES (1, ?, ?, ?, ?)
                   ON CONFLICT(singleton) DO UPDATE SET run_id = excluded.run_id,
                     gpu_id = excluded.gpu_id, owner = excluded.owner, updated_at = excluded.updated_at""",
                (run_id, gpu_id, owner, utc_now()),
            )

    def viewer(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT run_id, gpu_id, owner, updated_at FROM viewer_state WHERE singleton = 1").fetchone()
        return dict(row) if row else None
