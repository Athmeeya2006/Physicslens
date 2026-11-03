"""Does one servo calibration fit every SO-100? Each arm's identified laws replayed on every other arm's held-out episodes.

  python scripts/robot_fleet.py      # after robot_demo.py on the original arm and on each fleet arm (PL_ROBOT_FLEET)

For each arm (the original case and the other owners' arms in data/robot/fleet/), the winning servo law and its
parameters per joint (scripts/robot_demo.py) are run on the held-out episodes of every arm. The constant command
offset of each arm (the calibration between its leader and follower) is re-centred from that arm's own training
episodes, so what moves between arms is only the dynamics: lag, delay, deadband, damping. If one calibration fitted
every arm, the borrowed laws would track as well as each arm's own. The MuJoCo replay is not used here: the map from
each owner's logged degrees to the model's joints depends on that owner's calibration.
Local math, free. Writes data/robot/fleet.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.robot import servo as S

ARM = [0, 1, 2, 3, 4]
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
OUT = ROOT / "data" / "robot" / "fleet.json"


def arms() -> list[dict]:
    out = [{"id": "lerobot__svla_so100_pickplace", "result": ROOT / "data" / "robot" / "result.json",
            "data": [ROOT / "var" / "robot" / "raw" / "lerobot" / "data" / "chunk-000" / "file-000.parquet"]}]
    for d in sorted((ROOT / "data" / "robot" / "fleet").glob("*/result.json")):
        src = ROOT / "var" / "robot" / "fleet" / d.parent.name
        out.append({"id": d.parent.name, "result": d, "data": sorted((src / "data").rglob("episode_*.parquet"))})
    return out


def episodes(files: list[Path]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    us, ys = {}, {}
    for e, g in d.groupby("episode_index"):
        g = g.sort_values("frame_index")
        us[int(e)] = np.stack(g["action"].to_numpy()).astype(float)
        ys[int(e)] = np.stack(g["observation.state"].to_numpy()).astype(float)
    return us, ys


def arm_error(law: dict, us, ys, train: list[int], test: list[int], dt: float) -> np.ndarray:
    """Per held-out episode RMS over the five arm joints (deg) of the given laws, offsets re-centred on train."""
    per_joint = []
    for j, name in zip(ARM, NAMES):
        lid, p = law[name]
        tr = S.Batch.from_lists([us[e][:, j] for e in train], [ys[e][:, j] for e in train], dt, train)
        te = S.Batch.from_lists([us[e][:, j] for e in test], [ys[e][:, j] for e in test], dt, test)
        sim_tr = S.simulate(lid, p, tr)
        shift = float(np.sum((tr.y - sim_tr) * tr.mask) / np.sum(tr.mask))
        per_joint.append(S.rms_per_episode(S.simulate(lid, p, te) + shift, te))
    return np.sqrt(np.mean(np.square(per_joint), axis=0))


def main() -> None:
    fleet = arms()
    data, laws, meta = {}, {}, {}
    for a in fleet:
        r = json.loads(a["result"].read_text())
        laws[a["id"]] = {n: (r["joints"][n]["law"], {k: v["value"] for k, v in r["joints"][n]["params"].items()}) for n in NAMES}
        meta[a["id"]] = {"dataset": r["data"]["dataset"], "revision": r["data"]["revision"], "licence": r["data"]["licence"],
                         "episodes": r["data"]["episodes"], "train": r["data"]["train_episodes"], "test": r["data"]["test_episodes"],
                         "params": {n: {k: {"value": v["value"], "ci95": v["ci95"]} for k, v in r["joints"][n]["params"].items()}
                                    for n in NAMES},
                         "laws": {n: r["joints"][n]["law"] for n in NAMES}}
        data[a["id"]] = episodes(a["data"])
        print(a["id"], len(data[a["id"]][0]), "episodes", flush=True)
    dt = 1 / 30
    ids = [a["id"] for a in fleet]
    matrix = {}
    for target in ids:
        us, ys = data[target]
        own = arm_error(laws[target], us, ys, meta[target]["train"], meta[target]["test"], dt)
        row = {}
        for source in ids:
            e = own if source == target else arm_error(laws[source], us, ys, meta[target]["train"], meta[target]["test"], dt)
            mu, lo, hi = S.mean_interval(e)
            d_mu, d_lo, d_hi = S.mean_interval(e - own)
            row[source] = {"rms_deg": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)],
                           "minus_own_deg": round(d_mu, 4), "minus_own_ci95": [round(d_lo, 4), round(d_hi, 4)]}
        matrix[target] = row
        print(target, {s: row[s]["rms_deg"] for s in ids}, flush=True)
    # How much worse a borrowed calibration is than an arm's own, over every (arm, other arm) pair.
    worse = [matrix[t][s]["minus_own_deg"] for t in ids for s in ids if s != t]
    sig = [matrix[t][s]["minus_own_ci95"][0] > 0 for t in ids for s in ids if s != t]
    own_mean = float(np.mean([matrix[t][t]["rms_deg"] for t in ids]))
    borrowed_mean = float(np.mean([matrix[t][s]["rms_deg"] for t in ids for s in ids if s != t]))
    out = {"generated": "scripts/robot_fleet.py", "arms": meta, "matrix": matrix,
           "summary": {"arms": len(ids), "pairs": len(worse), "own_mean_deg": round(own_mean, 4),
                       "borrowed_mean_deg": round(borrowed_mean, 4), "ratio": round(borrowed_mean / own_mean, 3),
                       "pairs_significantly_worse": int(sum(sig)), "median_extra_deg": round(float(np.median(worse)), 4)},
           "method": ("each arm's winning servo law per joint (tau, delay, deadband, damping as fitted) run on every arm's "
                      "held-out episodes, the constant command offset re-centred from the target arm's training episodes; "
                      "per-episode RMS over the five arm joints, 95% bootstrap intervals over episodes")}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
