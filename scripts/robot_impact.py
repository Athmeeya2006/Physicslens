"""What the corrected SO-100 simulator changes in practice: where the gripper ends up, and a compensator built from it.

  python scripts/robot_impact.py     # after scripts/robot_demo.py (it saves the held-out MuJoCo replays)

1. Gripper position. Every held-out replay of scripts/robot_demo.py (the public model, the public model with a
   fitted delay and offsets, the corrected model) is turned into the position of the fixed jaw's tip by forward
   kinematics in the same MuJoCo model, and compared with the tip position of the measured joint angles. Reported:
   the RMS over each episode, and the error at the two moments a pick-and-place depends on: the pick (the gripper
   command, after opening, closes past half its opening) and the place (its widest opening after the pick).
2. Compensation. Each joint's identified law is inverted: the command that makes the servo follow a planned path
   (the path d seconds ahead, plus the lag or second-order drive the law needs, plus the deadband, minus the offset).
   Checked two ways. On real data: run backwards on the held-out episodes' measured motion, does it recover the
   command that was really sent? In simulation (the identified law standing in for the arm): sending a planned path
   as is against sending the compensated command.
Local math, free. Writes data/robot/impact.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import robot_demo as R
from physicslens.robot import arm_sim as A
from physicslens.robot import servo as S

OUT = ROOT / "data" / "robot" / "impact.json"
REPLAYS = R.CACHE.with_name(R.CACHE.stem.replace("identification", "replays") + ".npz")
MODELS = {"default": "public model, as published", "baseline_delay_offset": "public model + fitted delay + per-joint offsets",
          "identified": "corrected model"}


def tip_point(m: mujoco.MjModel) -> tuple[int, np.ndarray]:
    """The fixed jaw's body id and its tip in that body's frame: the mesh vertex farthest from the body origin."""
    b = m.body("Fixed_Jaw").id
    best, far = np.zeros(3), -1.0
    for g in np.flatnonzero(m.geom_bodyid == b):
        mid = m.geom_dataid[g]
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mid < 0:
            continue
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        pts = v @ rot.reshape(3, 3).T + m.geom_pos[g]
        k = int(np.argmax(np.linalg.norm(pts, axis=1)))
        if np.linalg.norm(pts[k]) > far:
            best, far = pts[k], float(np.linalg.norm(pts[k]))
    return b, best


def tip_path(m: mujoco.MjModel, jm: A.JointMap, body: int, local: np.ndarray, deg: np.ndarray) -> np.ndarray:
    """[N, 3] tip position (m) for joint angles [N, 6] in logged degrees."""
    d = mujoco.MjData(m)
    adr = [m.joint(n).qposadr[0] for n in jm.names]
    out = np.empty((len(deg), 3))
    for i, q in enumerate(jm.to_model(deg)):
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        out[i] = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local
    return out


def moments(u_grip: np.ndarray) -> tuple[int | None, int | None]:
    """Frame of the pick (after the first wide opening, the command closes past half of it) and of the place (the
    widest opening after the pick). None when an episode has no clear grasp."""
    peak = int(np.argmax(u_grip[: len(u_grip) * 2 // 3]))
    if u_grip[peak] < 10:
        return None, None
    below = np.flatnonzero(u_grip[peak:] < 0.5 * u_grip[peak])
    if not len(below):
        return None, None
    pick = peak + int(below[0])
    after = u_grip[pick:]
    place = pick + int(np.argmax(after)) if len(after) > 3 and after.max() > u_grip[pick] + 3 else None
    return pick, place


def summary(x: np.ndarray) -> dict:
    mu, lo, hi = S.mean_interval(np.asarray(x))
    return {"mean": round(mu, 2), "ci95": [round(lo, 2), round(hi, 2)], "median": round(float(np.median(x)), 2),
            "p90": round(float(np.percentile(x, 90)), 2), "n": len(x)}


# ---------- compensation ----------

def derivs(y: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and acceleration of a 30 fps encoder track (Savitzky-Golay, 7 frames, quadratic)."""
    return (savgol_filter(y, 7, 2, deriv=1, delta=dt, mode="interp"), savgol_filter(y, 7, 2, deriv=2, delta=dt, mode="interp"))


def compensate(law_id: str, p: dict[str, float], yd: np.ndarray, dt: float) -> np.ndarray:
    """The command [N] (deg) that makes this law follow the planned path yd [N]: the inverse of servo.simulate.
    The rate limit cannot be undone and is left as it is."""
    law = S.LAWS[law_id]
    f = set(law.features)
    v, a = derivs(yd, dt)
    if law.base == "lag1":
        e = p["tau"] * v
    else:
        drive = a + 2 * p["zeta"] * p["wn"] * v
        if "coulomb" in f:
            drive = drive + p["fc"] * np.tanh(v / S.SIGN_EPS)
        e = drive / p["wn"] ** 2
    if "deadband" in f:
        # Inside the band the servo does not move, so a planned move must step over it; standing still needs nothing.
        e = e + np.sign(e) * p["db"] * (np.abs(v) > 1.0)
    off = np.full_like(yd, p["b"])
    if "sag" in f:
        r = np.deg2rad(yd)
        off = off + p["sc"] * np.cos(r) + p["ss"] * np.sin(r)
    need = yd + e - off  # what u(t - d) must be at time t
    t = np.arange(len(yd)) * dt
    return np.interp(t + p["d"], t, need)  # u(t) = need(t + d): the plan is known d seconds ahead


def main() -> None:
    if not REPLAYS.exists():
        raise SystemExit(f"{REPLAYS.relative_to(ROOT)} is missing: run python scripts/robot_demo.py --reuse first")
    result = json.loads((R.OUT / "result.json").read_text())
    z = np.load(REPLAYS)
    m = A.load(R.XML)
    jm = R.joint_map()
    body, local = tip_point(m)
    test = R.TEST_EPS

    # 1. gripper position
    per_model: dict[str, dict[str, list[float]]] = {k: {"rms": [], "pick": [], "place": []} for k in MODELS}
    n_pick = n_place = 0
    for e in test:
        real = tip_path(m, jm, body, local, R.YS[e])
        pick, place = moments(R.US[e][:, 5])
        n_pick += pick is not None
        n_place += place is not None
        for k in MODELS:
            sim = z[f"{k}__{e}"].copy()
            sim[:, 5] = R.YS[e][:, 5]  # the jaw does not move the fixed jaw's tip; it is not simulated here
            err = np.linalg.norm(tip_path(m, jm, body, local, sim) - real, axis=1) * 1000
            per_model[k]["rms"].append(float(np.sqrt(np.mean(err ** 2))))
            if pick is not None:
                per_model[k]["pick"].append(float(err[pick]))
            if place is not None:
                per_model[k]["place"].append(float(err[place]))
    gripper = {k: {"label": MODELS[k], "rms_mm": summary(v["rms"]), "at_pick_mm": summary(v["pick"]),
                   "at_place_mm": summary(v["place"]),
                   "within_1cm_at_pick_and_place": round(float(np.mean(np.r_[v["pick"], v["place"]] <= 10.0)), 3)}
               for k, v in per_model.items()}
    pr = np.array(per_model["default"]["rms"]) / np.array(per_model["identified"]["rms"])
    print(json.dumps({k: (g["rms_mm"]["mean"], g["at_pick_mm"]["median"], g["at_place_mm"]["median"]) for k, g in gripper.items()}))

    # 2. compensation
    laws = {j: (result["joints"][n]["law"], {k: v["value"] for k, v in result["joints"][n]["params"].items()})
            for j, n in enumerate(R.LOG_JOINTS) if j in R.ARM}
    delay = result["fair_baseline"]["delay_s"]
    offsets = np.asarray(result["stronger_baseline"]["offsets_deg"])
    real_rows: dict[str, list[float]] = {"send_the_path": [], "delay_and_offset": [], "compensator": []}
    sim_rows: dict[str, list[float]] = {"send_the_path": [], "compensator": []}
    t = np.arange(max(len(R.US[e]) for e in test)) * R.DT
    for e in test:
        n = len(R.US[e])
        sq = {k: [] for k in real_rows}
        sq_sim = {k: [] for k in sim_rows}
        for j, (lid, p) in laws.items():
            y, u = R.YS[e][:, j], R.US[e][:, j]
            sq["send_the_path"].append(np.mean((y - u) ** 2))
            sq["delay_and_offset"].append(np.mean((np.interp(t[:n] + delay, t[:n], y) - offsets[j] - u) ** 2))
            sq["compensator"].append(np.mean((compensate(lid, p, y, R.DT) - u) ** 2))
            # Simulation: the planned path is a held-out leader path (smooth human motion); the identified law is the arm.
            plan = u
            for k, cmd in (("send_the_path", plan - p["b"]), ("compensator", compensate(lid, p, plan, R.DT))):
                b = S.Batch.from_lists([cmd], [plan], R.DT)
                sq_sim[k].append(np.mean((S.simulate(lid, p, b)[0] - plan) ** 2))
        for k, rows in real_rows.items():
            rows.append(float(np.sqrt(np.mean(sq[k]))))
        for k, rows in sim_rows.items():
            rows.append(float(np.sqrt(np.mean(sq_sim[k]))))
    comp = {"real_data": {"what": ("the command really sent on each held-out episode, predicted from the motion the arm "
                                   "made; RMS over the five arm joints, degrees"),
                          "rows": {k: summary(v) for k, v in real_rows.items()},
                          "labels": {"send_the_path": "assume the arm follows its command exactly",
                                     "delay_and_offset": "the fair baseline: one fitted delay and per-joint offsets",
                                     "compensator": "the identified laws inverted"}},
            "simulation": {"what": ("tracking error of a planned path (each held-out leader path) when the identified law "
                                    "stands in for the arm; RMS over the five arm joints, degrees"),
                           "rows": {k: summary(v) for k, v in sim_rows.items()},
                           "labels": {"send_the_path": "send the planned path (offset removed)",
                                      "compensator": "send the compensated command"}}}
    print(json.dumps({k: v["mean"] for k, v in comp["real_data"]["rows"].items()}),
          json.dumps({k: v["mean"] for k, v in comp["simulation"]["rows"].items()}))

    out = {"generated": "scripts/robot_impact.py", "dataset": result["data"]["dataset"], "test_episodes": test,
           "tip": {"body": "Fixed_Jaw", "point_in_body_m": [round(float(x), 4) for x in local],
                   "how": "the fixed jaw's mesh vertex farthest from its body origin; same joint map as the replays"},
           "episodes_with_pick": n_pick, "episodes_with_place": n_place,
           "moments": moments.__doc__.split("\n")[0] + " " + moments.__doc__.split("\n")[1].strip(),
           "gripper": gripper,
           "gripper_ratio_default_over_identified": summary(pr),
           "compensation": comp,
           "limits": [("the tip position uses the same assumed joint map as the replays (sign +1, rest pose shared); "
                       "the error is the simulator's, against the arm's own encoders, not against a camera"),
                      ("the compensator's real-data check reads the command back from motion that was made; a run on "
                       "the arm itself is the remaining test")]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
