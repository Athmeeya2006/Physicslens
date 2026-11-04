"""Does an SO-100's calibration go stale? One owner's arm, recorded in December 2024 and again in June 2025.

  python scripts/robot_drift.py      # after scripts/robot_demo.py on each recording below (PL_ROBOT_GROUP=drift)

Four public recordings by the same owner (pranavsaroha, Apache-2.0, single SO-100 arm, 30 fps): two a few days apart
in December 2024 and two a few days apart in June 2025. Each recording's identified servo laws are run on every other
recording's held-out episodes, as in scripts/robot_fleet.py (the constant command offset re-centred on the target's
own training episodes, so only the dynamics move). A calibration a few days old measures how well one calibration
repeats (the floor: different tasks, same arm); one six months old measures that plus what changed in between. If
the six-month penalty is no larger than the few-day one, there is no drift to see; if it is, the arm changed. That
the four recordings are one physical unit is the owner's account (one name, one robot type), not verified here.
Local math, free. Writes data/robot/drift.json.
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import robot_fleet as F
from physicslens.robot import servo as S

OUT = ROOT / "data" / "robot" / "drift.json"
# (dataset, recorded, where robot_demo.py wrote its result); medicine_0605 is also a fleet arm, so its result is reused.
RECORDINGS = [("pranavsaroha/so100_onelego3", "2024-12-26", "drift"),
              ("pranavsaroha/so100_squishy", "2024-12-30", "drift"),
              ("pranavsaroha/so100_medicine_0605", "2025-06-05", "fleet"),
              ("pranavsaroha/so100_med_cubes_0608", "2025-06-08", "drift")]
KEY = {"lag1": ("d", "tau", "db"), "so2": ("d", "wn", "db")}


def main() -> None:
    recs = []
    for repo, date, group in RECORDINGS:
        slug = repo.replace("/", "__")
        r = json.loads((ROOT / "data" / "robot" / group / slug / "result.json").read_text())
        files = sorted((ROOT / "var" / "robot" / group / slug / "data").rglob("episode_*.parquet"))
        recs.append({"id": slug, "dataset": repo, "recorded": date, "result": r, "data": F.episodes(files),
                     "laws": {n: (r["joints"][n]["law"], {k: v["value"] for k, v in r["joints"][n]["params"].items()})
                              for n in F.NAMES}})
        print(slug, len(recs[-1]["data"][0]), "episodes", flush=True)
    dt = 1 / 30
    own = {}
    for t in recs:
        us, ys = t["data"]
        own[t["id"]] = F.arm_error(t["laws"], us, ys, t["result"]["data"]["train_episodes"], t["result"]["data"]["test_episodes"], dt)
    pairs = []
    for s, t in permutations(recs, 2):
        us, ys = t["data"]
        e = F.arm_error(s["laws"], us, ys, t["result"]["data"]["train_episodes"], t["result"]["data"]["test_episodes"], dt)
        d_mu, d_lo, d_hi = S.mean_interval(e - own[t["id"]])
        gap = abs((np.datetime64(t["recorded"]) - np.datetime64(s["recorded"])).astype(int))
        pairs.append({"calibration_from": s["id"], "used_on": t["id"], "days_apart": int(gap),
                      "own_rms_deg": round(float(own[t["id"]].mean()), 4), "borrowed_rms_deg": round(float(e.mean()), 4),
                      "extra_deg": round(d_mu, 4), "extra_ci95": [round(d_lo, 4), round(d_hi, 4)]})
        print(pairs[-1], flush=True)
    near = [p["extra_deg"] for p in pairs if p["days_apart"] <= 10]
    far = [p["extra_deg"] for p in pairs if p["days_apart"] > 100]
    n_mu, n_lo, n_hi = S.mean_interval(np.array(near))
    f_mu, f_lo, f_hi = S.mean_interval(np.array(far))
    # Per joint: the delay, the lag (or natural frequency) and the deadband of each recording, with their intervals.
    params = {}
    for n in F.NAMES:
        params[n] = {}
        for rec in recs:
            j = rec["result"]["joints"][n]
            base = S.LAWS[j["law"]].base
            params[n][rec["id"]] = {"law": j["law"],
                                    **{k: {"value": j["params"][k]["value"], "ci95": j["params"][k]["ci95"]}
                                       for k in KEY[base] if k in j["params"]}}
    fleet = json.loads((ROOT / "data" / "robot" / "fleet.json").read_text())["summary"]
    out = {"generated": "scripts/robot_drift.py",
           "recordings": [{"id": r["id"], "dataset": r["dataset"], "recorded": r["recorded"],
                           "revision": r["result"]["data"]["revision"], "licence": r["result"]["data"]["licence"],
                           "episodes": r["result"]["data"]["episodes"], "own_rms_deg": round(float(own[r["id"]].mean()), 4)}
                          for r in recs],
           "pairs": pairs,
           "summary": {"days_apart_extra_deg": round(n_mu, 4), "days_apart_ci95": [round(n_lo, 4), round(n_hi, 4)],
                       "days_apart_pairs": len(near),
                       "six_months_extra_deg": round(f_mu, 4), "six_months_ci95": [round(f_lo, 4), round(f_hi, 4)],
                       "six_months_pairs": len(far),
                       "other_owners_median_extra_deg": fleet["median_extra_deg"],
                       "drift_seen": bool(f_lo > n_hi)},
           "params": params,
           "method": ("each recording's winning servo law per joint run on every other recording's held-out episodes, the "
                      "constant command offset re-centred from the target's training episodes; extra = borrowed minus own "
                      "per held-out episode, 95% bootstrap interval over episodes; the summary intervals are over pairs"),
           "limits": ["one owner, four recordings, different tasks; that they are one physical unit is the owner's naming",
                      "the summary intervals are over few pairs (4 near, 8 far), so they are wide"]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
