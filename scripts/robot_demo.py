"""Open-source robot arm: the SO-100 simulator corrected from public recordings (no hardware).

  python scripts/fetch_robot.py      download the pinned public files into var/robot/raw/ (free)
  python scripts/robot_demo.py       identify every joint, replay held-out episodes in MuJoCo, write data/robot/ (free)

Data: 50 teleoperated pick-and-place episodes of an SO-100 arm (LeRobot, lerobot/svla_so100_pickplace): per frame,
the command sent to the arm (the leader arm's reading, "action") and the arm's measured joint angles
("observation.state"), 30 frames per second. Model: MuJoCo Menagerie's trs_so_arm100.

Episodes are split by time, never shuffled: 0 to 24 fit every candidate law, 25 to 34 choose between them, the
winner is refitted on 0 to 34, and 35 to 49 are touched only once, for the final comparison. Local math only: no
model calls, no network (scripts/fetch_robot.py does the one download).
"""

from __future__ import annotations

import difflib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.robot import arm_sim as A
from physicslens.robot import servo as S

RAW = ROOT / "var" / "robot" / "raw"
XML = RAW / "menagerie" / "so_arm100.xml"
# Another owner's arm (scripts/robot_fleet.py): PL_ROBOT_FLEET names a folder written by scripts/fetch_robot_fleet.py;
# the same rules run on it and write data/robot/fleet/<folder>/. Unset: the original case, exactly as published.
FLEET = os.environ.get("PL_ROBOT_FLEET")
if FLEET:
    SRC = ROOT / "var" / "robot" / "fleet" / FLEET
    DATA_FILE = SRC / "meta" / "info.json"
    INFO_FILE = SRC / "meta" / "info.json"
    OUT = ROOT / "data" / "robot" / "fleet" / FLEET
    CACHE = ROOT / "var" / "robot" / f"identification_{FLEET}.pkl"
    SOURCE = json.loads((SRC / "source.json").read_text()) if (SRC / "source.json").exists() else {}
else:
    SRC = RAW / "lerobot"
    DATA_FILE = RAW / "lerobot" / "data" / "chunk-000" / "file-000.parquet"
    INFO_FILE = RAW / "lerobot" / "meta" / "info.json"
    OUT = ROOT / "data" / "robot"
    CACHE = ROOT / "var" / "robot" / "identification.pkl"
    SOURCE = {"dataset": "lerobot/svla_so100_pickplace", "revision": "728583b5eaf9e739a7f119e2def466fa1d552402",
              "licence": "Apache-2.0"}


def splits(n: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """By time, never shuffled: the first half fits, the next fifth chooses, the last 30% is held out (25/10/15 of 50)."""
    a, b = round(n * 0.5), round(n * 0.7)
    return list(range(a)), list(range(a, b)), list(range(b)), list(range(b, n))
LOG_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MJ_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
ARM = [0, 1, 2, 3, 4]  # the gripper closes on objects the simulator does not contain, so it is reported apart
BOOT_REPS = 64
DELAY_GRID = np.round(np.arange(0.0, 0.2001, 0.01), 3)
WORKERS = 16
LIMITS = [
    ("only the motions in these 50 pick-and-place episodes (speeds up to about 170 deg/s); payloads, wear, "
     "temperature and other units of the arm are not covered"),
    ("the measured angle is the arm's own encoder reading, so the identified delay includes the bus and logging "
     "latency; that is the delay a replay of logged commands needs"),
    ("the gripper closes on objects that the simulator scene does not contain, so it is reported but not counted in "
     "the arm total"),
    ("the map from logged degrees to model radians is an assumption (see model.joint_map); it only moves where "
     "gravity acts, and heldout_comparison.gravity_effect_on_default shows how much that matters"),
]


def load_episodes() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], float]:
    info = json.loads(INFO_FILE.read_text())
    if FLEET:  # LeRobot v2.1: one parquet per episode
        d = pd.concat([pd.read_parquet(p) for p in sorted((SRC / "data").rglob("episode_*.parquet"))], ignore_index=True)
    else:
        d = pd.read_parquet(DATA_FILE)
    dt = 1.0 / float(info["fps"])
    us, ys = {}, {}
    for e, g in d.groupby("episode_index"):
        g = g.sort_values("frame_index")
        ts = g["timestamp"].to_numpy()
        # A dropped or repeated frame would break the fixed time step every law assumes: refuse, don't paper over.
        if not np.allclose(np.diff(ts), dt, atol=1e-3):
            raise SystemExit(f"episode {e}: timestamps are not evenly spaced at {dt:.4f} s")
        us[int(e)] = np.stack(g["action"].to_numpy()).astype(float)
        ys[int(e)] = np.stack(g["observation.state"].to_numpy()).astype(float)
    return us, ys, dt


US, YS, DT = load_episodes() if DATA_FILE.exists() else ({}, {}, 1 / 30)
FIT_EPS, SELECT_EPS, TRAIN_EPS, TEST_EPS = splits(len(US)) if US else splits(50)


def batch(j: int, eps: list[int]) -> S.Batch:
    return S.Batch.from_lists([US[e][:, j] for e in eps], [YS[e][:, j] for e in eps], DT, eps)


# ---------- identification (each job is one process) ----------

def job_fit(args: tuple[int, str]) -> tuple[int, str, dict, float, int, float, list[float]]:
    j, law = args
    f = S.fit(law, batch(j, FIT_EPS))
    val = S.rms_per_episode(S.simulate(law, f.params, batch(j, SELECT_EPS)), batch(j, SELECT_EPS))
    return j, law, f.params, f.rss, f.n, f.bic, val.tolist()


def job_refit(args: tuple[int, str]) -> tuple[int, dict, list[float], list[float]]:
    j, law = args
    f = S.fit(law, batch(j, TRAIN_EPS), max_nfev=400)
    tr = S.rms_per_episode(S.simulate(law, f.params, batch(j, TRAIN_EPS)), batch(j, TRAIN_EPS))
    te = S.rms_per_episode(S.simulate(law, f.params, batch(j, TEST_EPS)), batch(j, TEST_EPS))
    return j, f.params, tr.tolist(), te.tolist()


def job_boot(args: tuple[int, str, dict, int]) -> tuple[int, list[list[float]]]:
    j, law, best, seed = args
    return j, S.bootstrap(law, batch(j, TRAIN_EPS), best, reps=BOOT_REPS // 8, seed=seed).tolist()


def job_episode(args: tuple[int, str, dict, int]) -> tuple[int, int, dict]:
    j, law, best, e = args
    return j, e, S.fit(law, batch(j, [e]), start=best, max_nfev=80).params


# ---------- MuJoCo replay ----------

def joint_map() -> A.JointMap:
    """Sign +1 and the rest pose shared: every episode starts with the arm folded at rest, which the Menagerie file
    names as its "rest" keyframe. The joints whose rest sits at a limit (shoulder lift, elbow, jaw) leave it in the
    same direction in the log and in the model, which supports sign +1 for them; for the others it is assumed."""
    m = A.load(XML)
    rest = np.mean([YS[e][0] for e in TRAIN_EPS], axis=0)
    q_rest = m.key("rest").qpos.copy()
    return A.JointMap(MJ_JOINTS, rest, q_rest, np.ones(6))


def correction(e: int, laws: dict[int, tuple[str, dict]], recentre: np.ndarray) -> np.ndarray:
    """[N, 6] command offset (deg) of the identified model for episode e; zero on joints it does not identify."""
    out = np.zeros_like(US[e])
    for j, (law, p) in laws.items():
        out[:, j] = A.command_correction(law, p, US[e][:, j]) - recentre[j]
    return out


def job_replay(args: tuple) -> tuple[str, int, list[list[float]]]:
    """args: label, episode, MJCF text (None = Menagerie as is), command shift (s), constant per-joint offset or None,
    gravity on, and the identified command correction (laws, recentre) or None."""
    label, e, xml_text, shift, offset, gravity, corr = args
    m = A.load(XML, xml_text)
    off = np.zeros(6) if offset is None else np.asarray(offset)
    if corr is not None:
        off = off + correction(e, *corr)
    sim = A.replay(m, joint_map(), US[e], YS[e][0], DT, ctrl_shift_s=shift, ctrl_offset_deg=off, gravity=gravity)
    return label, e, sim.tolist()


def run_replays(pool: ProcessPoolExecutor, jobs: list[tuple]) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for label, e, sim in pool.map(job_replay, jobs):
        out.setdefault(label, {})[e] = np.asarray(sim)
    return out


def episode_rms(sims: dict[int, np.ndarray], eps: list[int]) -> np.ndarray:
    """[E, 6] RMS error (deg) of each joint in each episode."""
    return np.array([np.sqrt(np.mean((sims[e] - YS[e]) ** 2, axis=0)) for e in eps])


def arm_total(r: np.ndarray) -> np.ndarray:
    """Per-episode RMS over the five arm joints together (root of the mean of their squared RMS)."""
    return np.sqrt(np.mean(r[:, ARM] ** 2, axis=1))


def summarize(r: np.ndarray) -> dict:
    out = {"joints": {}}
    for j, name in enumerate(LOG_JOINTS):
        mu, lo, hi = S.mean_interval(r[:, j])
        out["joints"][name] = {"rms_deg": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    mu, lo, hi = S.mean_interval(arm_total(r))
    out["arm_total"] = {"rms_deg": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Arm-total RMS of a minus b per held-out episode: mean, 95% interval, and the ratio of the means."""
    ta, tb = arm_total(a), arm_total(b)
    mu, lo, hi = S.mean_interval(ta - tb)
    return {"improvement_deg": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)], "ratio": round(ta.mean() / tb.mean(), 3)}


def identify(pool: ProcessPoolExecutor, t0: float) -> tuple:
    """Every law on every joint, the selection, the refit, the bootstrap and the per-episode fits."""
    # 1. every law on every joint: fit on 0..24, score on 25..34
    fits: dict[int, dict[str, S.Fit]] = {j: {} for j in range(6)}
    val: dict[int, dict[str, np.ndarray]] = {j: {} for j in range(6)}
    for j, law, p, rss, n, b, v in pool.map(job_fit, [(j, law) for j in range(6) for law in S.LAWS]):
        fits[j][law] = S.Fit(law, p, rss, n, b)
        val[j][law] = np.asarray(v)
    print(f"fitted {len(S.LAWS)} laws x 6 joints ({time.time() - t0:.0f} s)", flush=True)
    winners, steps = {}, {}
    for j in range(6):
        winners[j], steps[j] = S.select(fits[j], val[j])
        print(LOG_JOINTS[j], "->", winners[j], flush=True)

    # 2. refit the winner on 0..34; bootstrap over episodes; one fit per episode for consistency
    refit = {}
    for j, p, tr, te in pool.map(job_refit, [(j, winners[j]) for j in range(6)]):
        refit[j] = {"params": p, "train_rms": tr, "test_rms": te}
    boots: dict[int, list] = {j: [] for j in range(6)}
    for j, rows in pool.map(job_boot, [(j, winners[j], refit[j]["params"], s) for j in range(6) for s in range(8)]):
        boots[j].extend(rows)
    per_ep: dict[int, dict[int, dict]] = {j: {} for j in range(6)}
    for j, e, p in pool.map(job_episode, [(j, winners[j], refit[j]["params"], e) for j in range(6) for e in TRAIN_EPS]):
        per_ep[j][e] = p
    print(f"bootstrap and per-episode fits done ({time.time() - t0:.0f} s)", flush=True)

    return fits, val, winners, steps, refit, boots, per_ep


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit("the joint table is missing: run python scripts/fetch_robot.py (and fetch_robot_fleet.py) first")
    t0 = time.time()
    n_frames = {e: len(US[e]) for e in US}
    with ProcessPoolExecutor(WORKERS) as pool:
        if "--reuse" in sys.argv and CACHE.exists():
            # Only the MuJoCo stage reruns; the identification is read back from the last full run.
            ident = pickle.loads(CACHE.read_bytes())
        else:
            ident = identify(pool, t0)
            CACHE.write_bytes(pickle.dumps(ident))
        fits, val, winners, steps, refit, boots, per_ep = ident

        # 3. MuJoCo: the fair baseline's single delay is chosen on the training episodes only
        m0 = A.load(XML)
        jm = joint_map()
        grid = run_replays(pool, [(f"delay={s}", e, None, float(s), None, True, None) for s in DELAY_GRID for e in TRAIN_EPS])
        grid_err = {float(s): float(arm_total(episode_rms(grid[f"delay={s}"], TRAIN_EPS)).mean()) for s in DELAY_GRID}
        best_delay = min(grid_err, key=grid_err.get)
        offsets = np.mean(np.concatenate([YS[e] - US[e] for e in TRAIN_EPS]), axis=0)

        pose = np.median(np.concatenate([YS[e] for e in TRAIN_EPS]), axis=0)
        inertia = A.effective_inertia(m0, jm, pose)
        # The gripper is left at the Menagerie defaults: it closes on objects the scene does not contain, and its
        # best law sits at its parameter bounds, a sign that it describes the contact rather than the servo.
        gains = {MJ_JOINTS[j]: A.servo_gains(winners[j], refit[j]["params"], float(inertia[j])) for j in ARM}
        note = ("Servo gains, delays and joint friction of the five arm joints identified from "
                f"{SOURCE.get('dataset')} (episodes 0 to {TRAIN_EPS[-1]}) by PhysicsLens scripts/robot_demo.py; see "
                "data/robot/REPORT.md. Commands also need the per-joint offsets in data/robot/result.json "
                "(calibration between the leader and follower arms). The gripper keeps the original values.")
        ident_xml = A.write_identified(XML.read_text(), gains, note)
        laws = {j: (winners[j], refit[j]["params"]) for j in ARM}
        # The servo laws' offset b already contains the real arm's static sag under gravity, and MuJoCo adds its
        # own: the mean replay residual on the training episodes is taken out of the command once, so it is not
        # counted twice. Training episodes only; the test episodes are not looked at.
        first = run_replays(pool, [("ident0", e, ident_xml, 0.0, None, True, (laws, np.zeros(6))) for e in TRAIN_EPS])
        recentre = np.mean(np.concatenate([first["ident0"][e] - YS[e] for e in TRAIN_EPS]), axis=0)
        recentre[[j for j in range(6) if j not in ARM]] = 0.0

        jobs = []
        for e in TEST_EPS:
            jobs += [("default", e, None, 0.0, None, True, None),
                     ("default_no_gravity", e, None, 0.0, None, False, None),
                     ("baseline_delay", e, None, best_delay, None, True, None),
                     ("baseline_delay_offset", e, None, best_delay, offsets.tolist(), True, None),
                     ("identified_mjcf_only", e, ident_xml, 0.0, None, True, None),
                     ("identified_not_recentred", e, ident_xml, 0.0, None, True, (laws, np.zeros(6))),
                     ("identified", e, ident_xml, 0.0, None, True, (laws, recentre))]
        sims = run_replays(pool, jobs)
    print(f"MuJoCo replays done ({time.time() - t0:.0f} s)", flush=True)

    servo_test = np.array([[refit[j]["test_rms"][i] for j in range(6)] for i in range(len(TEST_EPS))])
    rows = {k: episode_rms(v, TEST_EPS) for k, v in sims.items()}
    rows["identified_servo_laws"] = servo_test

    ident, base, dflt = rows["identified"], rows["baseline_delay"], rows["default"]
    beats_default = paired(dflt, ident)
    beats_base = paired(base, ident)
    beats_base2 = paired(rows["baseline_delay_offset"], ident)
    proven = beats_default["ci95"][0] > 0 and beats_base["ci95"][0] > 0
    status = "proven" if proven else "partial"

    joints_out = {}
    for j, name in enumerate(LOG_JOINTS):
        law = S.LAWS[winners[j]]
        b = np.asarray(boots[j])
        params = {}
        for k, pname in enumerate(law.params):
            v = refit[j]["params"][pname]
            lo_b, hi_b = S.BOUNDS[pname]
            ep_vals = np.array([per_ep[j][e][pname] for e in TRAIN_EPS])
            params[pname] = {
                "value": round(v, 5), "ci95": [round(float(np.percentile(b[:, k], 2.5)), 5),
                                               round(float(np.percentile(b[:, k], 97.5)), 5)],
                "at_bound": bool(abs(v - lo_b) < 1e-3 * (hi_b - lo_b) or abs(v - hi_b) < 1e-3 * (hi_b - lo_b)),
                "per_episode_median": round(float(np.median(ep_vals)), 5),
                "per_episode_iqr": [round(float(np.percentile(ep_vals, 25)), 5), round(float(np.percentile(ep_vals, 75)), 5)],
            }
        joints_out[name] = {
            "mjcf_joint": MJ_JOINTS[j], "law": winners[j], "title": law.title, "params": params,
            "selection": [{"from": s.parent, "to": s.child, "heldout_gain_deg": round(s.gain[0], 4),
                           "ci95": [round(s.gain[1], 4), round(s.gain[2], 4)], "delta_bic": round(s.delta_bic, 1),
                           "accepted": s.accepted} for s in steps[j]],
            "candidates": {law_id: {"params": {k: round(v, 5) for k, v in fits[j][law_id].params.items()},
                                    "bic_fit_episodes": round(fits[j][law_id].bic, 1),
                                    "rms_fit_episodes_deg": round(float(np.sqrt(fits[j][law_id].rss / fits[j][law_id].n)), 4),
                                    "rms_select_episodes_deg": round(float(val[j][law_id].mean()), 4)}
                           for law_id in S.LAWS},
            "mjcf": ({k: round(v, 5) for k, v in gains[MJ_JOINTS[j]].items()}
                     | {"inertia_kg_m2": round(float(inertia[j]), 6), "command_recentre_deg": round(float(recentre[j]), 4)}
                     if j in ARM else "not written: Menagerie defaults kept (contact with objects; law at its bounds)"),
        }

    default_params = {
        "kp": float(m0.actuator_gainprm[0, 0]), "kv": [round(float(-x), 4) for x in m0.actuator_biasprm[:, 2]],
        "forcerange": m0.actuator_forcerange[0].tolist(), "armature": m0.dof_armature.tolist(),
        "frictionloss": m0.dof_frictionloss.tolist(), "damping": m0.dof_damping.tolist(), "delay": 0.0,
        "timestep": float(m0.opt.timestep)}
    result = {
        "generated": "scripts/robot_demo.py",
        "status": status,
        "status_rule": ("proven only if the identified model's held-out arm error is lower than both the default "
                        "model's and the fair baseline's with the 95% interval of each paired improvement above zero"),
        "data": {"dataset": SOURCE.get("dataset"), "revision": SOURCE.get("revision"),
                 "licence": SOURCE.get("licence"), "fps": 1 / DT, "episodes": len(US), "frames": int(sum(n_frames.values())),
                 "fit_episodes": FIT_EPS, "select_episodes": SELECT_EPS, "train_episodes": TRAIN_EPS,
                 "test_episodes": TEST_EPS, "units": "degrees as logged (LeRobot SO-100 calibration)"},
        "model": {"source": "MuJoCo Menagerie trs_so_arm100", "commit": "b0655057a9df2abe39ea7fc989914ba6f248301c",
                  "licence": "Apache-2.0", "default": default_params,
                  "joint_map": {"log_rest_deg": jm.deg_ref.round(3).tolist(), "model_rest_rad": jm.q_ref.tolist(),
                                "sign": jm.sign.tolist(), "assumption": joint_map.__doc__.split("\n")[0]}},
        "joints": joints_out,
        "identified_model": {"what": ("the identified MJCF (so_arm100_identified.xml) driven by the logged command "
                                      "plus each joint's command offset: b and, where chosen, the angle-dependent "
                                      "offset, minus the recentring measured on the training episodes"),
                             "joints": [LOG_JOINTS[j] for j in ARM]},
        "fair_baseline": {"what": "the default model with one command delay for all joints, chosen on the training "
                                  "episodes by MuJoCo replay", "delay_s": best_delay,
                          "training_arm_rms_by_delay": {str(k): round(v, 4) for k, v in grid_err.items()}},
        "stronger_baseline": {"what": "the fair baseline plus a constant per-joint offset (mean measured minus "
                                      "commanded angle on the training episodes)",
                              "offsets_deg": offsets.round(4).tolist()},
        "heldout": {k: summarize(v) for k, v in rows.items()},
        "heldout_comparison": {"default_minus_identified": beats_default,
                               "fair_baseline_minus_identified": beats_base,
                               "stronger_baseline_minus_identified": beats_base2,
                               "gravity_effect_on_default": paired(rows["default"], rows["default_no_gravity"])},
        "limits": LIMITS,
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    (OUT / "so_arm100_identified.xml").write_text(ident_xml)
    diff = difflib.unified_diff(XML.read_text().splitlines(keepends=True), ident_xml.splitlines(keepends=True),
                                "menagerie/trs_so_arm100/so_arm100.xml", "data/robot/so_arm100_identified.xml")
    (OUT / "so_arm100_identified.diff").write_text("".join(diff))
    print(json.dumps({"status": status, "heldout": {k: v["arm_total"] for k, v in result["heldout"].items()},
                      "vs_default": beats_default, "vs_fair": beats_base, "vs_stronger": beats_base2}, indent=1))


if __name__ == "__main__":
    main()
