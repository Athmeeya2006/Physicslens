"""Score PhysicsLens against ground truth written down before the analysis (see benchmark/PROTOCOL.md).

  python scripts/benchmark.py score --truths benchmark/truths.json --results-dir var/runs
      reads each clip's result.json (run ids listed in truths.json under "run_id") and writes
      benchmark/results.json and benchmark/RESULTS.md

Scoring is local and free. Running the analyses themselves is a separate, paid step (the web app, or
scripts/glimmer_batch.py), done with approval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Z95 = 1.96


def measured(result: dict, name: str) -> tuple[float, float] | None:
    """A quantity from a result: a fitted parameter, or the period (from the dominant frequency) when asked for."""
    for p in result.get("parameters", []):
        if p.get("name") == name and p.get("value") is not None:
            return float(p["value"]), float(p.get("std") or 0.0)
    if name == "period_s":
        f = (result.get("summary") or {}).get("dominant_frequency_hz")
        if f:
            return 1.0 / float(f), 0.0
    if name == "e" and result.get("bounces"):
        b = result["bounces"]
        return float(b["e_mean"]), float(b.get("e_std") or 0.0)
    return None


def score(truths: list[dict], results_dir: Path) -> list[dict]:
    rows = []
    for t in truths:
        path = results_dir / t["run_id"] / "result.json" if t.get("run_id") else None
        result = json.loads(path.read_text()) if path and path.is_file() else None
        for name, true_value in t.get("truth", {}).items():
            row = {"clip": t["clip"], "family": t.get("family"), "quantity": name, "truth": true_value,
                   "status": (result or {}).get("status", "not analyzed")}
            m = measured(result, name) if result else None
            if m is not None:
                value, std = m
                row.update(value=value, std=std, error_pct=100 * (value - true_value) / true_value,
                           covered=bool(std > 0 and abs(value - true_value) <= Z95 * std))
            rows.append(row)
    return rows


def markdown(rows: list[dict]) -> str:
    lines = ["# Benchmark results", "",
             ("Ground truth was measured and written down before each analysis (benchmark/PROTOCOL.md). Every clip is "
             "listed, including failures."), "",
             "| Clip | Quantity | Truth | PhysicsLens | Error | 95% interval contains truth | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        got = f"{r['value']:.4g} ± {r['std']:.2g}" if "value" in r else "none"
        err = f"{r['error_pct']:+.1f}%" if "error_pct" in r else ""
        cov = ("yes" if r.get("covered") else "no") if "covered" in r else ""
        lines.append(f"| {r['clip']} | {r['quantity']} | {r['truth']:.4g} | {got} | {err} | {cov} | {r['status']} |")
    measured_rows = [r for r in rows if "error_pct" in r]
    if measured_rows:
        mae = sum(abs(r["error_pct"]) for r in measured_rows) / len(measured_rows)
        cov = sum(bool(r.get("covered")) for r in measured_rows)
        lines += ["", (f"Mean absolute error {mae:.1f}% over {len(measured_rows)} measured quantities; the 95% interval "
                      f"contains the truth in {cov} of {len(measured_rows)}.")]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--truths", default=str(ROOT / "benchmark" / "truths.json"))
    s.add_argument("--results-dir", default=str(ROOT / "var" / "runs"))
    s.add_argument("--out", default=str(ROOT / "benchmark"))
    a = ap.parse_args()
    rows = score(json.loads(Path(a.truths).read_text()), Path(a.results_dir))
    out = Path(a.out)
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))
    (out / "RESULTS.md").write_text(markdown(rows))
    print(markdown(rows))


if __name__ == "__main__":
    main()
