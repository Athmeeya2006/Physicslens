"""The agent on the Sun's real data: Muse Spark Open proposes the law of individual sunspots' drift, and the pipeline's
verification decides, the same way it does for a video. PAID (Muse Spark Open on Baseten, a few cents per spot).

  python scripts/sun_agent.py [--spots 3]

Rebuilds the sunspot tracks from the cached SDO/HMI frames (scripts/sun_demo.py must have run once; Helioviewer is asked only for frames not cached), takes
the longest tracks sampled every 6 hours without gaps, and runs each spot's Carrington longitude against time through
astro/analysis.py's series mode. Each result is compared with the plain least-squares rate the /sun page reports, so
the agent's answer is checked by an independent calculation. Writes data/astro/sun/agent.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

PERIOD = "2014-01-01"  # the busiest of the five periods: large, long-lived spots
OUT = ROOT / "data" / "astro" / "sun" / "agent.json"


def regular(tr, step: float = 0.25) -> bool:
    t = np.array([s.t for s in tr.spots])
    return bool(len(t) >= 12 and np.all(np.diff(t) <= 2 * step + 1e-6))


def main() -> None:
    os.environ.setdefault("APP_ENV", "demo")  # open-weight reasoning only, as judges get it
    from physicslens.astro import sun as S
    from physicslens.astro.analysis import analyze_series
    from physicslens.costguard import CostGuardClient
    from physicslens.settings import load_settings
    from sun_demo import load_period

    n_spots = int(sys.argv[sys.argv.index("--spots") + 1]) if "--spots" in sys.argv else 3
    settings = load_settings()
    if settings.mock_models:
        sys.exit("Needs MOCK_MODELS=0: it calls Muse Spark Open for real.")
    client = CostGuardClient(settings)
    tracks = [tr for tr in S.usable(S.link(load_period(PERIOD, False))) if regular(tr)]
    tracks.sort(key=lambda tr: -len(tr.spots))
    rows = []
    for k, tr in enumerate(tracks[:n_spots]):
        t, lon_carr, _ = tr.arrays()
        ls = S.track_rates(tr)  # the page's own least-squares rate, the independent check
        name = f"a sunspot at latitude {tr.lat:+.1f} deg"
        t0 = time.time()
        r = analyze_series(client, t, lon_carr, run_id=f"sun-agent-{k}-{int(time.time())}", name=name,
                           value_unit="deg", time_unit="day", audience="dev",
                           provenance={"data_source": f"SDO/HMI continuum via Helioviewer, period from {PERIOD}, Carrington "
                                                      "longitude from each image's own header (sunpy)",
                                       "time_source": "image timestamps", "time_sigma_s": 1.0})
        # In the time-series family the drift is the motion's starting speed (part of the fitted state), not a law parameter.
        fs = r.get("fit_state") or {}
        row = {"spot": name, "lat": tr.lat, "points": len(t), "days": float(t[-1] - t[0]), "status": r["status"],
               "reason": r["reason"], "equation": r.get("equation"), "parameters": r.get("parameters"),
               "ranking": [{k2: f.get(k2) for k2 in ("model_id", "title", "heldout_nrmse", "bic", "status", "why")}
                           for f in r.get("ranking", [])],
               "trail": r.get("trail"), "llm_calls": r.get("llm_calls"),
               "least_squares": {"carrington_drift_deg_day": ls["sidereal"] - S.CARRINGTON_DEG_PER_DAY,
                                 "sigma": ls["sidereal_sigma"]}}
        if len(fs.get("y0") or []) >= 2:
            row["agent_drift_deg_day"] = float(fs["y0"][1])
        rows.append(row)
        print(f"{name}: {r['status']} {(r.get('equation') or {}).get('title')} | agent drift "
              f"{row.get('agent_drift_deg_day')} | least squares {row['least_squares']['carrington_drift_deg_day']:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    OUT.write_text(json.dumps({"generated": "scripts/sun_agent.py", "period": PERIOD, "spots": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
