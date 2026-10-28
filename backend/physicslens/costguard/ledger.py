"""Persistent SQLite spend ledger. Reservations are rows too, so a crash mid-call still counts."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    images INTEGER NOT NULL DEFAULT 0,
    video_frames INTEGER NOT NULL DEFAULT 0,
    audio_seconds REAL NOT NULL DEFAULT 0,
    gpu_seconds REAL NOT NULL DEFAULT 0,
    est_cost_usd REAL NOT NULL DEFAULT 0,
    run_id TEXT,
    status TEXT NOT NULL,
    audience TEXT NOT NULL DEFAULT 'dev',
    note TEXT
);
CREATE INDEX IF NOT EXISTS spend_day ON spend (substr(ts, 1, 10), provider);
CREATE INDEX IF NOT EXISTS spend_run ON spend (run_id);
"""

# A reservation left behind by a crash stays counted: the request may have been billed.
COUNTED_STATUSES = ("reserved", "committed", "failed_billed")


def utc_now() -> datetime:
    return datetime.now(UTC)


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE takes the write lock up front, so check-then-reserve is atomic across processes."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @staticmethod
    def spent(
        conn: sqlite3.Connection,
        *,
        provider: str | None = None,
        day: str | None = None,
        run_id: str | None = None,
        audience: str | None = None,
        model: str | None = None,
    ) -> float:
        where = [f"status IN ({','.join('?' * len(COUNTED_STATUSES))})"]
        args: list[Any] = list(COUNTED_STATUSES)
        for column, value in (("provider", provider), ("substr(ts, 1, 10)", day), ("run_id", run_id), ("audience", audience), ("model", model)):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        row = conn.execute(f"SELECT COALESCE(SUM(est_cost_usd), 0) FROM spend WHERE {' AND '.join(where)}", args).fetchone()
        return float(row[0])

    @staticmethod
    def insert(conn: sqlite3.Connection, **fields: Any) -> int:
        fields.setdefault("ts", utc_now().isoformat())
        cols = ", ".join(fields)
        marks = ", ".join("?" * len(fields))
        cur = conn.execute(f"INSERT INTO spend ({cols}) VALUES ({marks})", list(fields.values()))
        return int(cur.lastrowid)

    def update(self, row_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self.transaction() as conn:
            conn.execute(f"UPDATE spend SET {sets} WHERE id = ?", [*fields.values(), row_id])

    def rows(self) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT * FROM spend ORDER BY id").fetchall()
