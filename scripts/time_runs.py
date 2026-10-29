"""Time real analyses the way a judge experiences them: upload over HTTP, poll until the verdict.

  python scripts/time_runs.py --max-usd 1.40 CLIP[:pendulum_length_m] ...

Starts the backend locally in demo mode (reasoning on the open-weight provider in REASONING_PROVIDER, judge access
code from .env), runs the clips one after another, and writes var/timing/<stamp>.json with per-clip wall time,
verdict, the model behind each reasoning step and the ledger's spend. It SPENDS MONEY (SAM 3.1 and the reasoning
model), stops before the next clip once --max-usd is reached, and never re-runs a clip: a cached answer would be
a fake timing. The tunnel to the public site is not included; it adds well under a second per request.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8010
API = f"http://127.0.0.1:{PORT}"


def spent_by_provider() -> dict[str, float]:
    db = ROOT / "var" / "ledger.db"
    if not db.exists():
        return {}
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT provider, COALESCE(SUM(est_cost_usd), 0) FROM spend "
                            "WHERE status IN ('committed', 'failed_billed') GROUP BY provider").fetchall()
    return {p: float(v) for p, v in rows}


def post_clip(path: Path, code: str, fields: dict[str, str]) -> str:
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="{path.name}"\r\n'
                 f"Content-Type: video/mp4\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{API}/api/runs", data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}", "X-Access-Code": code})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["run_id"]


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="path, optionally :pendulum_length_m")
    ap.add_argument("--max-usd", type=float, required=True, help="stop before the next clip once spend reaches this")
    a = ap.parse_args()

    from dotenv import dotenv_values
    env_file = dotenv_values(ROOT / ".env")
    code = env_file.get("DEMO_ACCESS_CODE") or ""
    if not code:
        sys.exit("DEMO_ACCESS_CODE is not set in .env.")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "backend"), APP_ENV="demo", MOCK_MODELS="0", USE_CONTRIBUTOR_MODEL="0",
               REASONING_PROVIDER=os.environ.get("REASONING_PROVIDER", "baseten"), PUBLIC_TRY_ENABLED="0")
    out_dir = ROOT / "var" / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "backend.log", "a")  # noqa: SIM115 (open for the life of the backend)
    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "physicslens.app:app", "--host", "127.0.0.1",
                                "--port", str(PORT)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                if get("/api/health")["status"] == "ok":
                    break
            except OSError:
                time.sleep(1)
        start_spend = spent_by_provider()
        results = []
        for spec in a.clips:
            spent_now = sum(spent_by_provider().values()) - sum(start_spend.values())
            if spent_now >= a.max_usd:
                print(f"stopping: ${spent_now:.3f} spent, limit ${a.max_usd:.2f}", flush=True)
                break
            path_s, _, length = spec.partition(":")
            path = ROOT / path_s
            before = spent_by_provider()
            t0 = time.time()
            run_id = post_clip(path, code, {"pendulum_length_m": length} if length else {})
            while True:
                state = get(f"/api/runs/{run_id}")
                if state["status"] not in ("queued", "running"):
                    break
                time.sleep(1)
            seconds = time.time() - t0
            after = spent_by_provider()
            result = state.get("result") or {}
            calls = result.get("llm_calls") or []
            row = {
                "clip": path.name, "run_id": run_id, "seconds": round(seconds, 1), "status": state["status"],
                "verdict": result.get("status"), "family": result.get("family"),
                "equation": (result.get("equation") or {}).get("title"), "error": state.get("error"),
                "reasoning_calls": [{"step": c["step"], "provider": c["provider"], "model": c["model"], "cached": c["cached"]}
                                    for c in calls],
                "spend_usd": {p: round(after.get(p, 0) - before.get(p, 0), 5) for p in after
                              if after.get(p, 0) - before.get(p, 0) > 0},
            }
            results.append(row)
            print(f"{row['clip']}: {row['seconds']} s, {row['status']}/{row['verdict']} {row['equation'] or ''} "
                  f"calls={len(calls)} spend={row['spend_usd']}", flush=True)
        done = [r["seconds"] for r in results if r["status"] == "done"]
        summary = {"n": len(results), "completed": len(done),
                   "median_s": round(statistics.median(done), 1) if done else None,
                   "worst_s": max(done) if done else None,
                   "total_spend_usd": {p: round(v - start_spend.get(p, 0), 5) for p, v in spent_by_provider().items()
                                       if v - start_spend.get(p, 0) > 0}}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out_dir / f"{stamp}.json").write_text(json.dumps({"summary": summary, "runs": results}, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        backend.terminate()


if __name__ == "__main__":
    main()
