"""`python -m physicslens.costguard.report`: spend per provider/model/day and what is left."""

from __future__ import annotations

from collections import defaultdict

from ..settings import load_budget, load_settings
from .guard import Guard
from .ledger import COUNTED_STATUSES, Ledger


def render(ledger: Ledger, guard: Guard) -> str:
    rows = ledger.rows()
    counted = [r for r in rows if r["status"] in COUNTED_STATUSES]
    lines = [f"PhysicsLens spend report (ledger: {ledger.path})", ""]

    by_day: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for r in counted:
        key = (r["ts"][:10], r["provider"], r["model"])
        by_day[key][0] += r["est_cost_usd"]
        by_day[key][1] += 1
    lines.append(f"{'day (UTC)':<12} {'provider':<10} {'model':<34} {'calls':>5} {'cost USD':>10}")
    for (day, provider, model), (cost, n) in sorted(by_day.items()):
        lines.append(f"{day:<12} {provider:<10} {model:<34} {n:>5} {cost:>10.4f}")
    if not by_day:
        lines.append("(no spend recorded)")

    by_audience: dict[str, float] = defaultdict(float)
    for r in counted:
        by_audience[r["audience"]] += r["est_cost_usd"]
    lines += ["", "By audience: " + ", ".join(f"{a} ${by_audience.get(a, 0.0):.4f}" for a in ("dev", "judge", "public"))]

    lines += ["", f"{'provider':<10} {'spent':>10} {'hard cap':>10} {'left':>10} {'today':>10} {'daily cap':>10}"]
    for provider, s in guard.summary().items():
        left = s["hard_cap_usd"] - s["spent_usd"]
        lines.append(
            f"{provider:<10} {s['spent_usd']:>10.4f} {s['hard_cap_usd']:>10.2f} {left:>10.4f} "
            f"{s['spent_today_usd']:>10.4f} {s['daily_cap_usd']:>10.2f}"
        )

    open_res = sum(1 for r in rows if r["status"] == "reserved")
    hits = sum(1 for r in rows if r["status"] == "cache_hit")
    lines += ["", f"Cache hits (free): {hits}. Open reservations (counted as spent): {open_res}."]
    lines += [
        "",
        "REMINDER: this ledger is an estimate. The Meta dashboard Usage page is the truth, and it also includes",
        "playground use and other keys that the ledger cannot see. If the dashboard shows more, lower the caps",
        "in config/budget.yaml by the difference.",
    ]
    return "\n".join(lines)


def main() -> None:
    settings = load_settings()
    ledger = Ledger(settings.ledger_path)
    print(render(ledger, Guard(ledger, load_budget(settings.config_dir))))


if __name__ == "__main__":
    main()
