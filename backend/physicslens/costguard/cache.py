"""Response cache: re-processing the same clip or prompt costs nothing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    value TEXT NOT NULL,
    created REAL NOT NULL,
    expires REAL
);
"""


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_key(model: str, payload: Any, media: list[bytes | str] | None = None) -> str:
    """media: raw bytes, or digests already computed (e.g. a video hash, never a provider file_id)."""
    h = hashlib.sha256()
    h.update(json.dumps({"model": model, "payload": payload}, sort_keys=True, default=str).encode())
    for item in media or []:
        h.update(hashlib.sha256(item).digest() if isinstance(item, bytes) else item.encode())
    return h.hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as conn, conn:
            conn.executescript(SCHEMA)

    def get(self, key: str) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.path, timeout=30)) as conn:
            row = conn.execute("SELECT value, expires FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None or (row[1] is not None and row[1] < time.time()):
            return None
        return json.loads(row[0])

    def put(self, key: str, model: str, value: dict[str, Any], ttl_seconds: float | None = None) -> None:
        expires = time.time() + ttl_seconds if ttl_seconds else None
        with closing(sqlite3.connect(self.path, timeout=30)) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, model, value, created, expires) VALUES (?, ?, ?, ?, ?)",
                (key, model, json.dumps(value), time.time(), expires),
            )
