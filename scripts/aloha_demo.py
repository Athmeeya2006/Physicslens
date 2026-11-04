"""A second robot: the ALOHA arms' simulator, against a model that was already system-identified.

  python scripts/aloha_demo.py fetch     download the pinned public files into var/robot/aloha/ (free)
  python scripts/aloha_demo.py           identify every arm joint, replay held-out episodes in MuJoCo, write data/robot/aloha/
  python scripts/aloha_demo.py fetch-transfer && python scripts/aloha_demo.py transfer
                                         the same correction, unchanged, on the robot's other tasks (free)

Data: 50 teleoperated episodes of a real ALOHA (two ViperX arms; LeRobot, lerobot/aloha_static_coffee, MIT): per
frame, the leader arms' joint angles ("action", the command) and the follower arms' ("observation.state"), radians,
50 frames per second. Model: MuJoCo Menagerie's aloha, whose actuator gains, torque limits, joint damping, armature
and friction were already fitted by its authors to sinusoidal test motions on a real ALOHA (its README). So the
public model here is a dedicated identification, not a guess: the question is whether ordinary task recordings of
this unit, through the same servo laws and selection as the SO-100 case (scripts/robot_demo.py), track it better.

The twelve arm joints are identified; the grippers are not (their command and state are in different units).
Episodes are split by time, never shuffled: the first half fits, the next fifth chooses between laws, the last 30%
is touched once, for the final comparison. Local math only.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.robot import arm_sim as A
from physicslens.robot import servo as S

DATASET, REVISION = "lerobot/aloha_static_coffee", "b144896feb1f37398a862927b22cd3abdf005a6b"
MENAGERIE = "fb36886b3023db08c198264e1e9956ab202c3e3b"
RAW = ROOT / "var" / "robot" / "aloha"
XML = RAW / "menagerie" / "aloha" / "aloha.xml"
OUT = ROOT / "data" / "robot" / "aloha"
CACHE = ROOT / "var" / "robot" / "identification_aloha.pkl"  # the identification, so a later stage can rerun alone
MOTORS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# Log columns of the twelve arm joints (the grippers, columns 6 and 13, are left out) and their MJCF joint names.
COLS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
NAMES = [f"{side}/{m}" for side in ("left", "right") for m in MOTORS]
WORKERS = 28
BOOT_REPS = 32
DELAY_GRID = np.round(np.arange(0.0, 0.2001, 0.01), 3)


def fetch() -> None:
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
    for rel in ("meta/info.json", "data/chunk-000/file-000.parquet"):
        p = RAW / "lerobot" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{base}/{rel}", timeout=300) as r:
            p.write_bytes(r.read())
    with urllib.request.urlopen(f"https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{MENAGERIE}?recursive=1",
                                timeout=60) as r:
        tree = json.loads(r.read())["tree"]
    for item in tree:
        if item["type"] == "blob" and item["path"].startswith("aloha/"):
            p = RAW / "menagerie" / item["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE}/{item['path']}"
            with urllib.request.urlopen(url, timeout=120) as r:
                p.write_bytes(r.read())
    print("fetched", DATASET, REVISION[:10], "and menagerie aloha", MENAGERIE[:10])


def load_episodes() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], float]:
    info = json.loads((RAW / "lerobot" / "meta" / "info.json").read_text())
    d = pd.read_parquet(RAW / "lerobot" / "data" / "chunk-000" / "file-000.parquet")
    dt = 1.0 / float(info["fps"])
    us, ys = {}, {}
    for e, g in d.groupby("episode_index"):
        g = g.sort_values("frame_index")
        if not np.allclose(np.diff(g["timestamp"].to_numpy()), dt, atol=1e-3):
            raise SystemExit(f"episode {e}: timestamps are not evenly spaced at {dt:.4f} s")
        # Radians in the log; the servo laws and their bounds are in degrees.
        us[int(e)] = np.degrees(np.stack(g["action"].to_numpy()).astype(float)[:, COLS])
        ys[int(e)] = np.degrees(np.stack(g["observation.state"].to_numpy()).astype(float)[:, COLS])
    return us, ys, dt


HAVE = (RAW / "lerobot" / "data" / "chunk-000" / "file-000.parquet").exists()
US, YS, DT = load_episodes() if HAVE else ({}, {}, 1 / 50)
N = len(US) or 50
FIT_EPS, SELECT_EPS = list(range(round(N * 0.5))), list(range(round(N * 0.5), round(N * 0.7)))
TRAIN_EPS, TEST_EPS = list(range(round(N * 0.7))), list(range(round(N * 0.7), N))
J = len(COLS)


def batch(j: int, eps: list[int]) -> S.Batch:
    return S.Batch.from_lists([US[e][:, j] for e in eps], [YS[e][:, j] for e in eps], DT, eps)


def job_fit(args: tuple[int, str]) -> tuple[int, str, dict, float, int, float, list[float]]:
    j, law = args
    f = S.fit(law, batch(j, FIT_EPS))
    val = S.rms_per_episode(S.simulate(law, f.params, batch(j, SELECT_EPS)), batch(j, SELECT_EPS))
    return j, law, f.params, f.rss, f.n, f.bic, val.tolist()


def job_refit(args: tuple[int, str]) -> tuple[int, dict, list[float]]:
    j, law = args
    f = S.fit(law, batch(j, TRAIN_EPS), max_nfev=400)
    return j, f.params, S.rms_per_episode(S.simulate(law, f.params, batch(j, TEST_EPS)), batch(j, TEST_EPS)).tolist()


def job_boot(args: tuple[int, str, dict, int]) -> tuple[int, list[list[float]]]:
    j, law, best, seed = args
    return j, S.bootstrap(law, batch(j, TRAIN_EPS), best, reps=BOOT_REPS // 8, seed=seed).tolist()


# ---------- MuJoCo ----------

def spec(identified: dict[str, dict[str, float]] | None = None) -> mujoco.MjSpec:
    """The Menagerie model, with each arm actuator named after its joint (as arm_sim.replay expects); with
    `identified`, each actuator gets the identified kp, kv and delay, and each joint the identified dry friction in
    place of its damping and friction (the actuator's kv now carries the damping)."""
    s = mujoco.MjSpec.from_file(str(XML))
    ts = s.option.timestep
    for act in s.actuators:
        if act.target in NAMES:
            act.name = act.target
            if identified:
                g = identified[act.target]
                act.gainprm[0] = g["kp"]
                act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -g["kp"], -g["kv"]
                act.delay = g["delay"]
                act.nsample = max(2, math.ceil(g["delay"] / ts) + 2)
    if identified:
        for n, g in identified.items():
            jt = s.joint(n)
            jt.damping = [0.0, 0.0, 0.0]  # MuJoCo 3.17 keeps joint damping as a 3-vector
            jt.frictionloss = g["frictionloss"]
    return s


def joint_map() -> A.JointMap:
    """The log's radians are the model's: ALOHA's calibration zeroes each joint where the URDF, and the Menagerie
    model built from it, has its zero (an assumption, stated in the result)."""
    return A.JointMap(tuple(NAMES), np.zeros(J), np.zeros(J), np.ones(J))


def job_replay(args: tuple) -> tuple[str, int, list[list[float]]]:
    """label, episode, identified gains (None = Menagerie), command shift (s), per-joint offset or None, correction."""
    label, e, gains, shift, offset, corr = args
    m = spec(gains).compile()
    off = np.zeros(J) if offset is None else np.asarray(offset)
    if corr is not None:
        laws, recentre = corr
        c = np.zeros_like(US[e])
        for j, (law, p) in laws.items():
            c[:, j] = A.command_correction(law, p, US[e][:, j]) - recentre[j]
        off = off + c
    sim = A.replay(m, joint_map(), US[e], YS[e][0], DT, ctrl_shift_s=shift, ctrl_offset_deg=off)
    return label, e, sim.tolist()


def run_replays(pool: ProcessPoolExecutor, jobs: list[tuple]) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for label, e, sim in pool.map(job_replay, jobs):
        out.setdefault(label, {})[e] = np.asarray(sim)
    return out


def episode_rms(sims: dict[int, np.ndarray], eps: list[int]) -> np.ndarray:
    return np.array([np.sqrt(np.mean((sims[e] - YS[e]) ** 2, axis=0)) for e in eps])


def total(r: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(r ** 2, axis=1))


def summarize(r: np.ndarray) -> dict:
    out = {"joints": {}}
    for j, n in enumerate(NAMES):
        mu, lo, hi = S.mean_interval(r[:, j])
        out["joints"][n] = {"rms_deg": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    mu, lo, hi = S.mean_interval(total(r))
    out["arm_total"] = {"rms_deg": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    ta, tb = total(a), total(b)
    mu, lo, hi = S.mean_interval(ta - tb)
    return {"improvement_deg": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)], "ratio": round(ta.mean() / tb.mean(), 3)}


def main() -> None:
    if not HAVE:
        raise SystemExit("the joint table is missing: run python scripts/aloha_demo.py fetch first")
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as pool:
        if CACHE.exists() and "--reuse" in sys.argv:
            winners, steps, refit, servo_test, boots = pickle.loads(CACHE.read_bytes())
        else:
            winners, steps, refit, servo_test, boots = identify(pool, t0)
            CACHE.write_bytes(pickle.dumps((winners, steps, refit, servo_test, boots)))
        mujoco_stage(pool, t0, winners, steps, refit, servo_test, boots)


def identify(pool: ProcessPoolExecutor, t0: float) -> tuple:
    fits: dict[int, dict[str, S.Fit]] = {j: {} for j in range(J)}
    val: dict[int, dict[str, np.ndarray]] = {j: {} for j in range(J)}
    for j, law, p, rss, n, b, v in pool.map(job_fit, [(j, law) for j in range(J) for law in S.LAWS]):
        fits[j][law] = S.Fit(law, p, rss, n, b)
        val[j][law] = np.asarray(v)
    print(f"fitted {len(S.LAWS)} laws x {J} joints ({time.time() - t0:.0f} s)", flush=True)
    winners, steps = {}, {}
    for j in range(J):
        winners[j], steps[j] = S.select(fits[j], val[j])
        print(NAMES[j], "->", winners[j], flush=True)
    refit, servo_test = {}, {}
    for j, p, te in pool.map(job_refit, [(j, winners[j]) for j in range(J)]):
        refit[j], servo_test[j] = p, te
    boots: dict[int, list] = {j: [] for j in range(J)}
    for j, rows in pool.map(job_boot, [(j, winners[j], refit[j], s) for j in range(J) for s in range(8)]):
        boots[j].extend(rows)
    print(f"refit and bootstrap done ({time.time() - t0:.0f} s)", flush=True)
    return winners, steps, refit, servo_test, boots


def mujoco_stage(pool: ProcessPoolExecutor, t0: float, winners, steps, refit, servo_test, boots) -> None:
    m0 = spec().compile()
    jm = joint_map()
    grid = run_replays(pool, [(f"delay={s}", e, None, float(s), None, None) for s in DELAY_GRID for e in TRAIN_EPS])
    grid_err = {float(s): float(total(episode_rms(grid[f"delay={s}"], TRAIN_EPS)).mean()) for s in DELAY_GRID}
    best_delay = min(grid_err, key=grid_err.get)
    offsets = np.mean(np.concatenate([YS[e] - US[e] for e in TRAIN_EPS]), axis=0)
    pose = np.median(np.concatenate([YS[e] for e in TRAIN_EPS]), axis=0)
    inertia = A.effective_inertia(m0, jm, pose)
    gains = {NAMES[j]: A.servo_gains(winners[j], refit[j], float(inertia[j])) for j in range(J)}
    laws = {j: (winners[j], refit[j]) for j in range(J)}
    first = run_replays(pool, [("ident0", e, gains, 0.0, None, (laws, np.zeros(J))) for e in TRAIN_EPS])
    recentre = np.mean(np.concatenate([first["ident0"][e] - YS[e] for e in TRAIN_EPS]), axis=0)
    jobs = []
    for e in TEST_EPS:
        jobs += [("default", e, None, 0.0, None, None),
                 ("baseline_delay", e, None, best_delay, None, None),
                 ("baseline_delay_offset", e, None, best_delay, offsets.tolist(), None),
                 ("identified", e, gains, 0.0, None, (laws, recentre))]
    sims = run_replays(pool, jobs)
    print(f"MuJoCo replays done ({time.time() - t0:.0f} s)", flush=True)

    rows = {k: episode_rms(v, TEST_EPS) for k, v in sims.items()}
    rows["identified_servo_laws"] = np.array([[servo_test[j][i] for j in range(J)] for i in range(len(TEST_EPS))])
    vs_default = paired(rows["default"], rows["identified"])
    vs_fair = paired(rows["baseline_delay"], rows["identified"])
    vs_strong = paired(rows["baseline_delay_offset"], rows["identified"])
    status = "proven" if vs_default["ci95"][0] > 0 and vs_fair["ci95"][0] > 0 else "partial"
    joints = {}
    for j, n in enumerate(NAMES):
        law = S.LAWS[winners[j]]
        b = np.asarray(boots[j])
        joints[n] = {"law": winners[j], "title": law.title,
                     "params": {p: {"value": round(refit[j][p], 5),
                                    "ci95": [round(float(np.percentile(b[:, k], 2.5)), 5), round(float(np.percentile(b[:, k], 97.5)), 5)]}
                                for k, p in enumerate(law.params)},
                     "mjcf": {k: round(v, 5) for k, v in gains[n].items()} | {"command_recentre_deg": round(float(recentre[j]), 4)},
                     "menagerie": {"kp": float(m0.actuator_gainprm[m0.actuator(n).id, 0]),
                                   "damping": float(m0.dof_damping[m0.joint(n).dofadr[0]]),
                                   "frictionloss": float(m0.dof_frictionloss[m0.joint(n).dofadr[0]])}}
    result = {
        "generated": "scripts/aloha_demo.py", "status": status,
        "status_rule": ("proven only if the identified model's held-out error is lower than both the Menagerie model's and "
                        "the fair baseline's, with the 95% interval of each paired improvement above zero"),
        "data": {"dataset": DATASET, "revision": REVISION, "licence": "MIT", "fps": 1 / DT, "episodes": N,
                 "frames": int(sum(len(u) for u in US.values())), "fit_episodes": FIT_EPS, "select_episodes": SELECT_EPS,
                 "train_episodes": TRAIN_EPS, "test_episodes": TEST_EPS, "units": "degrees (logged in radians)"},
        "model": {"source": "MuJoCo Menagerie aloha", "commit": MENAGERIE, "licence": "BSD-3-Clause",
                  "already_identified": ("its authors fitted actuator gains, torque limits, joint damping, armature and "
                                         "friction to 11 sinusoidal trajectories on a real ALOHA (Menagerie aloha README)"),
                  "joint_map": joint_map.__doc__.split("\n")[0]},
        "joints": joints,
        "fair_baseline": {"what": "the Menagerie model with one command delay for all joints, chosen on the training episodes",
                          "delay_s": best_delay, "training_rms_by_delay": {str(k): round(v, 4) for k, v in grid_err.items()}},
        "stronger_baseline": {"what": "the fair baseline plus a constant per-joint offset (mean measured minus commanded)",
                              "offsets_deg": offsets.round(4).tolist()},
        "heldout": {k: summarize(v) for k, v in rows.items()},
        "heldout_comparison": {"default_minus_identified": vs_default, "fair_baseline_minus_identified": vs_fair,
                               "stronger_baseline_minus_identified": vs_strong},
        "limits": ["one ALOHA unit and one task (coffee); the Menagerie model was identified on another unit",
                   "the grippers are not identified (command and state are in different units)",
                   "the measured angle is the arm's own encoder reading; the delay includes bus and logging latency"],
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    # The Menagerie model is BSD-3-Clause (Trossen Robotics): the derived file keeps its notice and licence beside it.
    note = ("<!-- Derived from MuJoCo Menagerie's aloha model (commit fb36886b30), Copyright (c) 2023, Trossen Robotics, "
            "BSD-3-Clause; see LICENSE-menagerie-aloha beside this file. Changed by scripts/aloha_demo.py: each arm actuator's "
            "kp, kv and delay, and each arm joint's frictionloss in place of its damping, identified from lerobot/aloha_static_coffee. "
            "Commands also need the per-joint offsets in result.json. -->\n")
    (OUT / "aloha_identified.xml").write_text(note + spec(gains).to_xml())
    (OUT / "LICENSE-menagerie-aloha").write_text((XML.parent / "LICENSE").read_text())
    print(json.dumps({"status": status, "heldout": {k: v["arm_total"] for k, v in result["heldout"].items()},
                      "vs_default": vs_default, "vs_fair": vs_fair, "vs_stronger": vs_strong}, indent=1))


# ---------- transfer: the coffee-task correction, unchanged, on the same robot's other tasks ----------

TRANSFER = ["lerobot/aloha_static_battery", "lerobot/aloha_static_candy", "lerobot/aloha_static_tape"]


def fetch_transfer() -> None:
    for repo in TRANSFER:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=60) as r:
            meta = json.loads(r.read())
        rev = meta["sha"]
        licence = next((x.split(":", 1)[1] for x in meta.get("tags", []) if x.startswith("license:")), None)
        out = RAW / "transfer" / repo.split("/")[1]
        for rel in ("meta/info.json", "data/chunk-000/file-000.parquet"):
            f = out / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{rel}", timeout=300) as r:
                f.write_bytes(r.read())
        (out / "source.json").write_text(json.dumps({"dataset": repo, "revision": rev, "licence": licence}))
        print("fetched", repo, rev[:10], licence, flush=True)


def job_transfer(args: tuple) -> tuple[str, int, list[list[float]]]:
    label, e, gains, u, y0, corr = args
    m = spec(gains).compile()
    off = np.zeros(J)
    if corr is not None:
        laws, recentre = corr
        off = np.column_stack([A.command_correction(laws[j][0], laws[j][1], u[:, j]) - recentre[j] for j in range(J)])
    return label, e, A.replay(m, joint_map(), u, y0, DT, ctrl_offset_deg=off).tolist()


def transfer() -> None:
    """Every episode of each other task, replayed with the published model and with the coffee-task correction,
    nothing refitted (gains, laws and the command recentring all as identified on coffee)."""
    r = json.loads((OUT / "result.json").read_text())
    gains = {n: {k: r["joints"][n]["mjcf"][k] for k in ("kp", "kv", "delay", "frictionloss")} for n in NAMES}
    laws = {j: (r["joints"][n]["law"], {k: v["value"] for k, v in r["joints"][n]["params"].items()}) for j, n in enumerate(NAMES)}
    recentre = np.array([r["joints"][n]["mjcf"]["command_recentre_deg"] for n in NAMES])
    out = {"generated": "scripts/aloha_demo.py transfer", "identified_on": DATASET, "tasks": []}
    with ProcessPoolExecutor(WORKERS) as pool:
        for repo in TRANSFER:
            src = RAW / "transfer" / repo.split("/")[1]
            d = pd.read_parquet(src / "data" / "chunk-000" / "file-000.parquet")
            us, ys = {}, {}
            for e, g in d.groupby("episode_index"):
                g = g.sort_values("frame_index")
                us[int(e)] = np.degrees(np.stack(g["action"].to_numpy()).astype(float)[:, COLS])
                ys[int(e)] = np.degrees(np.stack(g["observation.state"].to_numpy()).astype(float)[:, COLS])
            eps = sorted(us)
            jobs = [(k, e, gn, us[e], ys[e][0], c) for e in eps
                    for k, gn, c in (("default", None, None), ("identified", gains, (laws, recentre)))]
            sims: dict[str, dict[int, np.ndarray]] = {}
            for label, e, sim in pool.map(job_transfer, jobs):
                sims.setdefault(label, {})[e] = np.asarray(sim)
            rows = {k: np.array([np.sqrt(np.mean((v[e] - ys[e]) ** 2, axis=0)) for e in eps]) for k, v in sims.items()}
            src_meta = json.loads((src / "source.json").read_text())
            task = {**src_meta, "episodes": len(eps), "default": summarize(rows["default"])["arm_total"],
                    "identified": summarize(rows["identified"])["arm_total"],
                    "default_minus_identified": paired(rows["default"], rows["identified"]),
                    "episodes_better": int(np.sum(total(rows["identified"]) < total(rows["default"])))}
            out["tasks"].append(task)
            print(repo, task["default"]["rms_deg"], task["identified"]["rms_deg"], task["default_minus_identified"], flush=True)
    (OUT / "transfer.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    elif sys.argv[1:] == ["fetch-transfer"]:
        fetch_transfer()
    elif sys.argv[1:] == ["transfer"]:
        transfer()
    else:
        main()
