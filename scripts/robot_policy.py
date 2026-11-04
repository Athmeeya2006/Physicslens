"""Calibrated on a human, tested on an AI: does an SO-100's identified model predict the arm under a learned policy?

  python scripts/robot_policy.py      # after robot_demo.py on the fleet arms; fetch the rollouts first with
                                      # PL_ROBOT_GROUP=policy python scripts/fetch_robot_fleet.py <dataset>

Every calibration in this repository was fitted on teleoperation: a person moving a leader arm. Three of the fleet's
owners also published their arm driven by a trained ACT policy (LeRobot eval_act_* datasets, Apache-2.0): the
"action" column is the policy's command, the state is where the arm went. On each rollout, from the policy's commands
alone, three predictions of the arm's joints are compared: the arm's own servo laws (fitted on its teleoperation,
never refitted), the fair baseline (one command delay and per-joint offsets, fitted on the same teleoperation
episodes), and every other arm's laws (a borrowed calibration). The evaluator is the fleet comparison's
(scripts/robot_fleet.py): the servo laws themselves, constant command offsets re-centred on the arm's own
teleoperation training episodes. The MuJoCo replay is not used: each owner's logged degrees map onto the model's
joints through that owner's own calibration. Local math, free. Writes data/robot/policy.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import robot_fleet as F
from physicslens.robot import servo as S

OUT = ROOT / "data" / "robot" / "policy.json"
DT = 1 / 30
# Fleet arm -> its owner's published ACT policy rollouts.
POLICIES = {"00ri__so100_battery": ["00ri__eval_act_so100_battery_bin_center"],
            "julyfun__so100_pick_and_place_v4_50": ["julyfun__eval_act_so100_test3"],
            "sergiov2000__so100_lego_and_ball_40_lego": ["sergiov2000__eval_act_so100_wm_lb_ball_horiz_2",
                                                         "sergiov2000__eval_act_so100_wm_lb_ball_vert_2",
                                                         "sergiov2000__eval_act_so100_wm_lb_lego_horiz_2",
                                                         "sergiov2000__eval_act_so100_wm_lb_lego_vert_2"]}
POLICY_ID0 = 100000  # rollouts join the arm's episode dict under ids that cannot collide with its own
DELAYS = np.round(np.arange(0.0, 0.2001, 1 / 30), 4)


def delay_offset(us, ys, train: list[int], test: list[int]) -> tuple[np.ndarray, float]:
    """The fair baseline: the arm follows its command a fixed delay later, plus a constant offset per joint, both
    fitted on the training episodes. Per held-out episode RMS over the five arm joints, and the delay."""
    def predict(e: int, d: float, off: np.ndarray) -> np.ndarray:
        t = np.arange(len(us[e])) * DT
        return np.column_stack([np.interp(t - d, t, us[e][:, j]) for j in F.ARM]) + off

    best = None
    for d in DELAYS:
        res = np.concatenate([ys[e][:, F.ARM] - predict(e, d, 0.0) for e in train])
        off = res.mean(axis=0)
        cost = float(np.mean((res - off) ** 2))
        if best is None or cost < best[0]:
            best = (cost, float(d), off)
    _, d, off = best
    return np.array([np.sqrt(np.mean((ys[e][:, F.ARM] - predict(e, d, off)) ** 2)) for e in test]), d


def iv(x: np.ndarray) -> dict:
    mu, lo, hi = S.mean_interval(np.asarray(x))
    return {"mean": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def main() -> None:
    fleet = {a["id"]: a for a in F.arms()}
    laws, meta = {}, {}
    for aid, a in fleet.items():
        r = json.loads(a["result"].read_text())
        laws[aid] = {n: (r["joints"][n]["law"], {k: v["value"] for k, v in r["joints"][n]["params"].items()}) for n in F.NAMES}
        meta[aid] = r["data"]
    arms_out, rows = [], []
    for aid, slugs in POLICIES.items():
        us, ys = F.episodes(fleet[aid]["data"])
        train, test = meta[aid]["train_episodes"], meta[aid]["test_episodes"]
        sources, pol = [], []
        for slug in slugs:
            src = ROOT / "var" / "robot" / "policy" / slug
            pu, py = F.episodes(sorted((src / "data").rglob("episode_*.parquet")))
            for e in sorted(pu):
                k = POLICY_ID0 + len(pol)
                us[k], ys[k] = pu[e], py[e]
                pol.append({"dataset": slug.replace("__", "/"), "episode": e, "frames": len(pu[e])})
            sources.append({"dataset": slug.replace("__", "/"),
                            **{k: v for k, v in json.loads((src / "source.json").read_text()).items() if k in ("revision", "licence", "episodes")}})
        pids = [POLICY_ID0 + i for i in range(len(pol))]
        own_tele = F.arm_error(laws[aid], us, ys, train, test, DT)
        own_pol = F.arm_error(laws[aid], us, ys, train, pids, DT)
        base_tele, d = delay_offset(us, ys, train, test)
        base_pol, _ = delay_offset(us, ys, train, pids)
        borrowed = np.array([F.arm_error(laws[s], us, ys, train, pids, DT) for s in laws if s != aid])  # [arms-1, rollouts]
        for i, p in enumerate(pol):
            rows.append({"arm": aid, **p, "own_deg": round(float(own_pol[i]), 4), "baseline_deg": round(float(base_pol[i]), 4),
                         "borrowed_median_deg": round(float(np.median(borrowed[:, i])), 4)})
        arms_out.append({"arm": aid, "calibration_dataset": meta[aid]["dataset"], "calibration_revision": meta[aid]["revision"],
                         "policy_datasets": sources, "rollouts": len(pol), "baseline_delay_s": d,
                         "teleop_heldout": {"own": iv(own_tele), "baseline": iv(base_tele)},
                         "policy": {"own": iv(own_pol), "baseline": iv(base_pol), "borrowed_median": iv(np.median(borrowed, axis=0)),
                                    "own_minus_baseline": iv(own_pol - base_pol)}})
        print(aid, json.dumps(arms_out[-1]["teleop_heldout"]), json.dumps(arms_out[-1]["policy"]), flush=True)
    # Calibrated on the policy's own motion: where an arm's rollouts were also calibrated by robot_demo.py
    # (PL_ROBOT_GROUP=policy), both calibrations predict that calibration's held-out rollouts, each with its constant
    # offset re-centred on the same policy training rollouts.
    retrained = []
    for aid, slugs in POLICIES.items():
        res = ROOT / "data" / "robot" / "policy" / slugs[0] / "result.json"
        if len(slugs) != 1 or not res.exists():
            continue
        pr = json.loads(res.read_text())
        plaws = {n: (pr["joints"][n]["law"], {k: v["value"] for k, v in pr["joints"][n]["params"].items()}) for n in F.NAMES}
        src = ROOT / "var" / "robot" / "policy" / slugs[0]
        pu, py = F.episodes(sorted((src / "data").rglob("episode_*.parquet")))
        tr, te = pr["data"]["train_episodes"], pr["data"]["test_episodes"]
        tele = F.arm_error(laws[aid], pu, py, tr, te, DT)
        pol_cal = F.arm_error(plaws, pu, py, tr, te, DT)
        retrained.append({"arm": aid, "policy_dataset": slugs[0].replace("__", "/"), "train_rollouts": len(tr), "test_rollouts": len(te),
                          "teleop_calibration": iv(tele), "policy_calibration": iv(pol_cal),
                          "policy_minus_teleop": iv(pol_cal - tele)})
        print("retrained", aid, json.dumps(retrained[-1]), flush=True)
    own = np.array([r["own_deg"] for r in rows])
    base = np.array([r["baseline_deg"] for r in rows])
    bor = np.array([r["borrowed_median_deg"] for r in rows])
    mu, lo, hi = S.mean_interval(base - own)
    mu2, lo2, hi2 = S.mean_interval(bor - own)
    out = {"generated": "scripts/robot_policy.py", "arms": arms_out, "rollouts": rows, "calibrated_on_policy": retrained,
           "summary": {"arms": len(arms_out), "rollouts": len(rows), "own_mean_deg": round(float(own.mean()), 4),
                       "baseline_mean_deg": round(float(base.mean()), 4), "borrowed_mean_deg": round(float(bor.mean()), 4),
                       "baseline_over_own": round(float(base.mean() / own.mean()), 3),
                       "baseline_minus_own_deg": round(mu, 4), "baseline_minus_own_ci95": [round(lo, 4), round(hi, 4)],
                       "borrowed_minus_own_deg": round(mu2, 4), "borrowed_minus_own_ci95": [round(lo2, 4), round(hi2, 4)],
                       "rollouts_own_better_than_baseline": int(np.sum(own < base)),
                       "rollouts_own_better_than_borrowed": int(np.sum(own < bor))},
           "method": ("each arm's servo laws from its own teleoperation (scripts/robot_demo.py, never refitted) run open loop "
                      "on its owner's AI-policy rollouts from the policy's commands; the fair baseline (one delay and "
                      "per-joint offsets) fitted on the same teleoperation training episodes; a borrowed calibration is "
                      "every other fleet arm's laws (median over them); RMS over the five arm joints per rollout; "
                      "intervals by bootstrap over rollouts"),
           "limits": [(f"{len(arms_out)} arms and {len(rows)} rollouts; the policies were recorded 1 to 15 days after the "
                       "teleoperation, on the same or a related task"),
                      "this tests the model's prediction of the arm under a policy's commands, not a policy trained in simulation"]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
