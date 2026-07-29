"""SQLite persistence for the single-instance VBOGS web scheduler."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted"})
DOWNLOAD_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})
DOWNLOAD_ACTIVE_STATUSES = frozenset({"queued", "running"})


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
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    revision INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS ncore_downloads (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS ncore_download_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    download_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY(download_id) REFERENCES ncore_downloads(id)
                );
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(viewer_state)")}
            for name, definition in (
                ("status", "TEXT NOT NULL DEFAULT 'idle'"),
                ("revision", "INTEGER NOT NULL DEFAULT 0"),
                ("error", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(f"ALTER TABLE viewer_state ADD COLUMN {name} {definition}")

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

    def list_runs(self, limit: int = 100, *, statuses: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """List newest runs, optionally limited to an explicit status set."""
        if statuses is not None and not statuses:
            return []
        query = "SELECT * FROM runs"
        values: list[Any] = []
        if statuses is not None:
            query += f" WHERE status IN ({', '.join('?' for _ in statuses)})"
            values.extend(statuses)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
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

    def delete_run(self, run_id: str, *, allowed_statuses: frozenset[str]) -> dict[str, Any] | None:
        """Delete a run and its event history if it remains in an allowed state."""
        with self._transaction() as connection:
            record = self._run(connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())
            if record is None or record["status"] not in allowed_statuses:
                return None
            connection.execute("DELETE FROM run_events WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return record

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

    def set_viewer(self, run_id: str, gpu_id: str, owner: str) -> dict[str, Any]:
        with self._transaction() as connection:
            row = connection.execute("SELECT revision FROM viewer_state WHERE singleton = 1").fetchone()
            revision = int(row["revision"]) + 1 if row else 1
            connection.execute(
                """INSERT INTO viewer_state(singleton, run_id, gpu_id, owner, updated_at, status, revision, error)
                   VALUES (1, ?, ?, ?, ?, 'active', ?, NULL)
                   ON CONFLICT(singleton) DO UPDATE SET run_id = excluded.run_id,
                     gpu_id = excluded.gpu_id, owner = excluded.owner, updated_at = excluded.updated_at,
                     status = excluded.status, revision = excluded.revision, error = NULL""",
                (run_id, gpu_id, owner, utc_now(), revision),
            )
        return self.viewer() or {}

    def clear_viewer(self) -> dict[str, Any]:
        with self._transaction() as connection:
            row = connection.execute("SELECT revision FROM viewer_state WHERE singleton = 1").fetchone()
            revision = int(row["revision"]) + 1 if row else 1
            connection.execute(
                """INSERT INTO viewer_state(singleton, run_id, gpu_id, owner, updated_at, status, revision, error)
                   VALUES (1, NULL, NULL, NULL, ?, 'idle', ?, NULL)
                   ON CONFLICT(singleton) DO UPDATE SET run_id = NULL, gpu_id = NULL, owner = NULL,
                     updated_at = excluded.updated_at, status = excluded.status,
                     revision = excluded.revision, error = NULL""",
                (utc_now(), revision),
            )
        return self.viewer() or {}

    def viewer(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_id, gpu_id, owner, updated_at, status, revision, error FROM viewer_state WHERE singleton = 1"
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _download(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def create_download(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._transaction() as connection:
            connection.execute(
                """INSERT INTO ncore_downloads (id, owner, scene_id, status, created_at)
                   VALUES (?, ?, ?, 'queued', ?)""",
                (record["id"], record["owner"], record["scene_id"], record["created_at"]),
            )
        self.add_download_event(record["id"], "Queued NCore download.")
        return self.get_download(record["id"]) or record

    def get_download(self, download_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._download(connection.execute(
                "SELECT * FROM ncore_downloads WHERE id = ?", (download_id,)
            ).fetchone())

    def list_downloads(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ncore_downloads ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._download(row) for row in rows if row is not None]

    def active_download_for_scene(self, scene_id: str) -> dict[str, Any] | None:
        placeholders = ", ".join("?" for _ in DOWNLOAD_ACTIVE_STATUSES)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM ncore_downloads WHERE scene_id = ? AND status IN ({placeholders}) "
                "ORDER BY created_at DESC LIMIT 1",
                [scene_id, *DOWNLOAD_ACTIVE_STATUSES],
            ).fetchone()
        return self._download(row)

    def next_queued_download(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ncore_downloads WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
        return self._download(row)

    def active_download(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ncore_downloads WHERE status = 'running' ORDER BY started_at ASC LIMIT 1"
            ).fetchone()
        return self._download(row)

    def transition_download(self, download_id: str, status: str, *, error: str | None = None) -> None:
        fields = ["status = ?"]
        values: list[Any] = [status]
        if status == "running":
            fields.append("started_at = COALESCE(started_at, ?)")
            values.append(utc_now())
        if status in DOWNLOAD_TERMINAL_STATUSES:
            fields.append("finished_at = ?")
            values.append(utc_now())
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        values.append(download_id)
        with self._transaction() as connection:
            connection.execute(f"UPDATE ncore_downloads SET {', '.join(fields)} WHERE id = ?", values)

    def add_download_event(self, download_id: str, message: str) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO ncore_download_events(download_id, created_at, message) VALUES (?, ?, ?)",
                (download_id, utc_now(), message),
            )
            return int(cursor.lastrowid)

    def download_events(self, download_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, created_at, message FROM ncore_download_events WHERE download_id = ? "
                "ORDER BY sequence DESC LIMIT ?", (download_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def mark_active_downloads_interrupted(self) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE ncore_downloads SET status = 'interrupted', finished_at = ?, error = ? "
                "WHERE status = 'running'",
                (utc_now(), "Web service restarted; resubmit to resume completed components."),
            )
        return cursor.rowcount
