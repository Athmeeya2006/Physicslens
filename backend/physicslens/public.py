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


def client_address(remote: str | None, forwarded_for: str | None, cf_connecting_ip: str | None = None) -> str:
    # Behind a proxy every request arrives from the proxy (a Cloudflare tunnel: 127.0.0.1), so the visitor's address
    # comes from a header; only trust headers when told to, otherwise anyone could rotate one to dodge the limit.
    # Cloudflare's own CF-Connecting-IP is set at its edge and can't be supplied by the visitor.
    if os.environ.get("TRUST_PROXY", "0") == "1":
        if cf_connecting_ip:
            return cf_connecting_ip.strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return remote or "unknown"


class VisitorAllowance:
    """Open judging (no access code): how many paid actions of each kind one visitor may start in the whole period.

    The money is capped by the judges' total allowance; this only keeps one visitor (or a bot) from using all of it
    before the judges arrive. Counted per address and kind, not per day."""

    def __init__(self, path: Path, limits: dict[str, int]):
        self.path = path
        self.limits = limits
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS visitor_use (addr TEXT, kind TEXT, n INTEGER, PRIMARY KEY (addr, kind))")

    def take(self, addr: str, kind: str) -> bool:
        limit = self.limits.get(kind, 0)
        with closing(sqlite3.connect(self.path, timeout=30, isolation_level=None)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT n FROM visitor_use WHERE addr = ? AND kind = ?", (addr, kind)).fetchone()
            used = row[0] if row else 0
            if used >= limit:
                conn.execute("ROLLBACK")
                return False
            conn.execute("INSERT OR REPLACE INTO visitor_use (addr, kind, n) VALUES (?, ?, ?)", (addr, kind, used + 1))
            conn.execute("COMMIT")
        return True


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
