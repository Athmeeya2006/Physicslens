"""Budget caps: reserve the worst-case cost before a call, settle it after."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .errors import BudgetExceeded
from .ledger import Ledger, utc_now
from .pricing import Usage

log = logging.getLogger("physicslens.costguard")

AUDIENCES = ("dev", "judge", "public")
# Shown to judges and public visitors instead of any quota wording.
BUSY_MESSAGE = "PhysicsLens is temporarily busy, please retry later."


@dataclass
class Reservation:
    id: int
    provider: str
    model: str
    run_id: str
    amount: float
    _guard: Guard = field(repr=False)
    settled: bool = False

    def commit(self, actual_cost: float, usage: Usage | None = None, status: str = "committed", note: str | None = None) -> None:
        self._guard.settle(self, actual_cost, usage, status, note)

    def release(self) -> None:
        self._guard.settle(self, 0.0, None, "released", None)


class Guard:
    def __init__(self, ledger: Ledger, budget: dict[str, Any], now: Callable[[], datetime] = utc_now):
        self.ledger = ledger
        self.budget = budget
        self._now = now
        self._lock = threading.Lock()

    def _caps(self, provider: str) -> dict[str, float]:
        caps = self.budget.get(provider)
        if not caps or "hard_cap_usd" not in caps or "daily_cap_usd" not in caps:
            raise BudgetExceeded(f"No budget caps for provider {provider!r} in config/budget.yaml; refusing the call.")
        return caps

    def check_and_reserve(
        self,
        provider: str,
        model: str,
        est_cost: float,
        run_id: str,
        audience: str = "dev",
        run_cap_usd: float | None = None,
    ) -> Reservation:
        if not math.isfinite(est_cost) or est_cost < 0:
            raise ValueError(f"est_cost must be a non-negative number, got {est_cost!r}")
        if audience not in AUDIENCES:
            raise ValueError(f"audience must be one of {AUDIENCES}")
        caps = self._caps(provider)
        run_cap = run_cap_usd if run_cap_usd is not None else float(self.budget["per_run_cap_usd"])
        today = self._now().date().isoformat()

        # The thread lock covers this process; BEGIN IMMEDIATE covers other processes.
        with self._lock, self.ledger.transaction() as conn:
            total = Ledger.spent(conn, provider=provider)
            daily = Ledger.spent(conn, provider=provider, day=today)
            run = Ledger.spent(conn, run_id=run_id)
            checks = [
                (total, caps["hard_cap_usd"], f"{provider} total budget"),
                (daily, caps["daily_cap_usd"], f"{provider} daily budget"),
                (run, run_cap, f"per-run budget for run {run_id}"),
            ]
            audience_cap = self.budget.get(f"{audience}_daily_cap_usd") if audience != "dev" else None
            if audience_cap is not None:
                checks.append((Ledger.spent(conn, day=today, audience=audience), audience_cap, f"{audience} daily safety cap"))
            model_cap = (self.budget.get("model_daily_caps_usd") or {}).get(model)
            if model_cap is not None:
                # Some models are approved for one narrow use with their own daily cap inside the provider's.
                checks.append((Ledger.spent(conn, day=today, model=model), float(model_cap), f"{model} daily budget"))

            for spent, cap, label in checks:
                if spent + est_cost > cap:
                    if audience != "dev":
                        log.error("SAFETY CAP TRIPPED: %s (spent %.4f + est %.4f > cap %.2f)", label, spent, est_cost, cap)
                        raise BudgetExceeded(BUSY_MESSAGE)
                    raise BudgetExceeded(
                        f"Refused: {label} would be exceeded (spent ${spent:.4f} + estimate ${est_cost:.4f} > cap ${cap:.2f})."
                    )

            warn = float(self.budget.get("warn_at_fraction", 0.75))
            for spent, cap, label in checks:
                if cap > 0 and spent < warn * cap <= spent + est_cost:
                    log.warning("Budget warning: %s passed %d%% (cap $%.2f).", label, warn * 100, cap)

            row_id = Ledger.insert(
                conn, provider=provider, model=model, est_cost_usd=est_cost, run_id=run_id,
                status="reserved", audience=audience, ts=self._now().isoformat(),
            )
        return Reservation(row_id, provider, model, run_id, est_cost, self)

    def settle(self, res: Reservation, actual_cost: float, usage: Usage | None, status: str, note: str | None) -> None:
        if res.settled:
            return
        if actual_cost > res.amount * 1.001 and status != "released":
            log.warning(
                "%s/%s cost $%.5f exceeded its reservation $%.5f; check prices.yaml and output limits.",
                res.provider, res.model, actual_cost, res.amount,
            )
        fields: dict[str, Any] = {"est_cost_usd": actual_cost, "status": status, "note": note}
        if usage is not None:
            fields.update(usage.as_dict())
        with self._lock:
            self.ledger.update(res.id, **fields)
        res.settled = True

    def summary(self) -> dict[str, Any]:
        """Spent vs caps per provider, for the dev UI header and the report."""
        today = self._now().date().isoformat()
        out: dict[str, Any] = {}
        with self.ledger.transaction() as conn:
            for provider, caps in self.budget.items():
                if not isinstance(caps, dict):
                    continue
                out[provider] = {
                    "spent_usd": round(Ledger.spent(conn, provider=provider), 6),
                    "hard_cap_usd": caps["hard_cap_usd"],
                    "spent_today_usd": round(Ledger.spent(conn, provider=provider, day=today), 6),
                    "daily_cap_usd": caps["daily_cap_usd"],
                }
        return out
