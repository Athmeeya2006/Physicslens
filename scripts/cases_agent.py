"""The real-world cases through the agent: Muse Spark Open proposes the laws, verification decides, as for a video.

  MOCK_MODELS=1 python scripts/cases_agent.py [CASE ...]   # free: the test doubles answer (checks the plumbing only)
  python scripts/cases_agent.py [CASE ...]                 # PAID: Muse Spark Open on Baseten, a few cents per case

CASES: wasp12b (transit-time residuals against time), kepler88b (the same for Kepler-88 b), fireball (Winchcombe's
distance along its track against time). The Sun's single spots are scripts/sun_agent.py.

Each case already has laws a person chose (scripts/timing_demo.py, scripts/fireball_demo.py). Here the agent sees only
the numbers and a one-line description, proposes its own laws (free-form ones included), and the pipeline's held-out
verification ranks them. The result sits beside the person's: which won, and whether the agent proposed something the
hand-chosen set lacked that forecasts better. Writes data/cases_agent/<case>.json. Mock runs write
data/cases_agent/mock/ instead, so a plumbing check can never be mistaken for a real result.
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

OUT = ROOT / "data" / "cases_agent"


def wasp12b() -> dict:
    t = json.loads((ROOT / "data/astro/timing/results.json").read_text())["wasp12b"]
    oc = [p for p in t["oc"] if not p[4]]  # transits only: occultations carry the opposite sign under precession
    bjd = np.array([p[1] for p in oc])
    return {"t": bjd - bjd.min(), "y": np.array([p[2] for p in oc]), "name": "WASP-12b's transit times minus a constant-period prediction",
            "value_unit": "min", "time_unit": "day",
            "provenance": {"data_source": "158 published mid-times, Yee et al. 2020 (ApJL 888, L5)", "time_source": "BJD_TDB",
                           "time_sigma_s": 60.0},
            "hand": [{"law": x["law"], "title": x["title"], "heldout_chi2": x["heldout_chi2"]} for x in t["laws"]],
            "hand_winner": t.get("winner")}


def kepler88b() -> dict:
    t = json.loads((ROOT / "data/astro/timing/results.json").read_text())["kepler88b"]
    bjd = np.array([p[1] for p in t["oc"]])
    return {"t": bjd - bjd.min(), "y": np.array([p[2] for p in t["oc"]]), "name": "Kepler-88 b's transit times minus a constant-period prediction",
            "value_unit": "min", "time_unit": "day",
            "provenance": {"data_source": "74 transit times measured from Kepler light curves (scripts/timing_demo.py)",
                           "time_source": "BJD_TDB", "time_sigma_s": 180.0},
            "hand": [{"law": x["law"], "title": x["title"], "heldout_chi2": x["heldout_chi2"]} for x in t["laws"]]
                    + ([{"law": "nbody", "title": "N-body: a second planet", "heldout_rms_min": t["companion"]["heldout_rms_min"]}]
                       if t.get("companion") else [])}


def fireball() -> dict:
    from fireball_demo import read_points
    tt, _h, d = read_points()
    # Several stations measure the same moments: average them in 0.05 s bins so the series is one value per time.
    bins = np.round(tt / 0.05).astype(int)
    keys = np.unique(bins)
    tb = np.array([tt[bins == k].mean() for k in keys])
    db = np.array([d[bins == k].mean() for k in keys])
    f = json.loads((ROOT / "data/astro/fireball/results.json").read_text())
    return {"t": tb - tb.min(), "y": (db - db.min()) / 1000.0, "name": "the Winchcombe fireball's distance along its track",
            "value_unit": "km", "time_unit": "s",
            "provenance": {"data_source": "published triangulated trajectory (McMullan et al. 2023, WMPL report), all stations",
                           "time_source": "station clocks", "time_sigma_s": 0.01},
            "hand": [{"law": "single_body", "title": "single-body ablation and drag (hand-chosen)",
                      "heldout_rms_kms": f["single_body"]["heldout_rms_kms"]},
                     {"law": "alpha_beta", "title": "alpha-beta (hand-chosen)", "heldout_rms_kms": f["alpha_beta"]["heldout_rms_kms"]}]}


CASES = {"wasp12b": wasp12b, "kepler88b": kepler88b, "fireball": fireball}


def run(case: str, client, mock: bool) -> dict:
    from physicslens.astro.analysis import analyze_series

    d = CASES[case]()
    t0 = time.time()
    r = analyze_series(client, d["t"], d["y"], run_id=f"cases-agent-{case}-{int(time.time())}", name=d["name"],
                       value_unit=d["value_unit"], time_unit=d["time_unit"], audience="dev", provenance=d["provenance"])
    rows = [{k: f.get(k) for k in ("model_id", "title", "heldout_nrmse", "bic", "status", "why")} for f in r.get("ranking", [])]
    out = {"generated": "scripts/cases_agent.py", "case": case, "mock": mock, "points": len(d["t"]),
           "status": r["status"], "reason": r["reason"], "equation": r.get("equation"), "parameters": r.get("parameters"),
           "agent_ranking": rows, "agent_proposed_free_form": [x["title"] for x in rows if str(x.get("model_id", "")).startswith("custom_")],
           "hand_chosen": d["hand"], "trail": r.get("trail"), "llm_calls": r.get("llm_calls"), "seconds": time.time() - t0}
    return out


def main() -> None:
    from physicslens.costguard import CostGuardClient
    from physicslens.settings import load_settings

    os.environ.setdefault("APP_ENV", "demo")  # open-weight reasoning only, as judges get it
    settings = load_settings()
    mock = settings.mock_models
    client = CostGuardClient(settings)
    cases = [a for a in sys.argv[1:] if a in CASES] or list(CASES)
    dest = OUT / "mock" if mock else OUT
    dest.mkdir(parents=True, exist_ok=True)
    for c in cases:
        out = run(c, client, mock)
        (dest / f"{c}.json").write_text(json.dumps(out, indent=1, default=float))
        print(f"{c}: {out['status']} {(out['equation'] or {}).get('title')} | free-form proposed: {out['agent_proposed_free_form']} "
              f"({out['seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
