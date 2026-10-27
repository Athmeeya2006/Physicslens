"""Limits for public visitors (no access code): a few free analyses per address per day, plus optional bot check."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import httpx

log = logging.getLogger("physicslens.public")

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


class PublicQuota:
    """Counts analyses per client address per UTC day. Judges and dev never pass through here."""

    def __init__(self, path: Path, per_day: int):
        self.path = path
        self.per_day = per_day
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS public_use (addr TEXT, day TEXT, n INTEGER, PRIMARY KEY (addr, day))")

    def take(self, addr: str) -> bool:
        """Reserve one analysis for this address today; False when today's allowance is used up."""
        day = datetime.now(UTC).date().isoformat()
        with closing(sqlite3.connect(self.path, timeout=30, isolation_level=None)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT n FROM public_use WHERE addr = ? AND day = ?", (addr, day)).fetchone()
            used = row[0] if row else 0
            if used >= self.per_day:
                conn.execute("ROLLBACK")
                return False
            conn.execute("INSERT OR REPLACE INTO public_use (addr, day, n) VALUES (?, ?, ?)", (addr, day, used + 1))
            conn.execute("COMMIT")
        return True


def client_address(remote: str | None, forwarded_for: str | None) -> str:
    # Behind a proxy the visitor's address is the first X-Forwarded-For entry; only trust it when told to,
    # otherwise anyone could rotate the header to dodge the limit.
    if forwarded_for and os.environ.get("TRUST_PROXY", "0") == "1":
        return forwarded_for.split(",")[0].strip()
    return remote or "unknown"


def turnstile_ok(token: str | None, addr: str) -> bool:
    """Cloudflare Turnstile check for public uploads (a free service, not a paid API). Off unless configured."""
    secret = os.environ.get("TURNSTILE_SECRET_KEY", "").strip()
    if not secret:
        return True
    if not token:
        return False
    try:
        r = httpx.post(TURNSTILE_VERIFY_URL, data={"secret": secret, "response": token, "remoteip": addr}, timeout=10)
        return bool(r.json().get("success"))
    except (httpx.HTTPError, ValueError) as e:
        log.warning("Turnstile verification failed: %s", e)
        return False
