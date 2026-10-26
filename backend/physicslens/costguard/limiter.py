"""Our own rate limits (far below the provider's) and a per-provider circuit breaker."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

from .errors import CircuitOpen, CostGuardError

log = logging.getLogger("physicslens.costguard")

DEFAULT_RPM = 10
# Slow down when the provider says fewer than this fraction of requests/tokens remain.
LOW_REMAINING_FRACTION = 0.1


class TokenBucket:
    def __init__(self, rpm: int, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.rate = rpm / 60.0
        # Small burst so a 30 RPM cap can't fire 30 requests in one second.
        self.capacity = float(max(1, min(rpm, 5)))
        self.tokens = self.capacity
        self.updated = clock()
        self.paused_until = 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def acquire(self, timeout: float) -> None:
        deadline = self._clock() + timeout
        while True:
            with self._lock:
                now = self._clock()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if now >= self.paused_until and self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = max(self.paused_until - now, (1 - self.tokens) / self.rate, 0.01)
            if self._clock() + wait > deadline:
                raise CostGuardError("Rate limit: waited too long for a request slot; try again shortly.")
            self._sleep(wait)

    def pause(self, seconds: float) -> None:
        with self._lock:
            self.paused_until = max(self.paused_until, self._clock() + seconds)


class RateLimiter:
    def __init__(self, limits: dict[str, Any], clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._rpm = limits.get("requests_per_minute") or {}
        self._clock = clock
        self._sleep = sleep
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._voice = threading.BoundedSemaphore(int(limits.get("voice_max_concurrent_streams", 2)))

    def bucket(self, model: str) -> TokenBucket:
        with self._lock:
            if model not in self._buckets:
                rpm = self._rpm.get(model)
                if rpm is None:
                    log.warning("No RPM cap for %s in config/limits.yaml; using %d.", model, DEFAULT_RPM)
                    rpm = DEFAULT_RPM
                self._buckets[model] = TokenBucket(int(rpm), self._clock, self._sleep)
            return self._buckets[model]

    def acquire(self, model: str, timeout: float = 60.0) -> None:
        self.bucket(model).acquire(timeout)

    def pause(self, model: str, seconds: float) -> None:
        self.bucket(model).pause(seconds)

    def observe_headers(self, model: str, headers: Mapping[str, str]) -> None:
        for kind in ("requests", "tokens"):
            remaining = _num(headers.get(f"x-ratelimit-remaining-{kind}"))
            limit = _num(headers.get(f"x-ratelimit-limit-{kind}"))
            if remaining is not None and limit and remaining / limit < LOW_REMAINING_FRACTION:
                log.warning("%s: only %d of %d %s left in the provider window; slowing down.", model, remaining, limit, kind)
                self.pause(model, 5.0)

    @contextmanager
    def voice_slot(self):
        if not self._voice.acquire(blocking=False):
            raise CostGuardError("Voice: too many concurrent streams; finish the current question first.")
        try:
            yield
        finally:
            self._voice.release()


class CircuitBreaker:
    def __init__(self, threshold: int = 5, pause_seconds: float = 120, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.pause_seconds = pause_seconds
        self._clock = clock
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, provider: str) -> None:
        with self._lock:
            remaining = self._open_until.get(provider, 0) - self._clock()
        if remaining > 0:
            raise CircuitOpen(f"{provider} is paused after repeated failures; retrying in {remaining:.0f} s.")

    def record_success(self, provider: str) -> None:
        with self._lock:
            self._failures[provider] = 0

    def record_failure(self, provider: str) -> None:
        with self._lock:
            n = self._failures.get(provider, 0) + 1
            self._failures[provider] = n
            if n >= self.threshold:
                self._open_until[provider] = self._clock() + self.pause_seconds
                self._failures[provider] = 0
                log.error("%s failed %d times in a row; pausing it for %.0f s.", provider, n, self.pause_seconds)


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
