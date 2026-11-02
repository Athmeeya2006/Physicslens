"""Open-loop replay of logged joint commands in MuJoCo, and writing identified servo laws into an MJCF.

Local and free: MuJoCo only. The logged arm reports joints in degrees with its own zero; the MJCF uses radians with
another zero. The map between them (sign +1, offset fixed by the rest pose) is an assumption, stated in the report;
it only moves where gravity acts, and the runner measures how much that matters by replaying without gravity.
"""

from __future__ import annotations

import copy
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .servo import LAWS


@dataclass(frozen=True)
class JointMap:
    """Degrees in the log to radians in the model: q = sign * deg2rad(deg - deg_ref) + q_ref."""
    names: tuple[str, ...]  # MJCF joint (and actuator) names, in log order
    deg_ref: np.ndarray
    q_ref: np.ndarray
    sign: np.ndarray

    def to_model(self, deg: np.ndarray) -> np.ndarray:
        return self.sign * np.deg2rad(deg - self.deg_ref) + self.q_ref

    def to_log(self, q: np.ndarray) -> np.ndarray:
        return np.rad2deg((q - self.q_ref) / self.sign) + self.deg_ref


def load(xml_path: Path, xml_text: str | None = None) -> mujoco.MjModel:
    """Load from a path, or from text resolved against the path's folder (so mesh paths still work)."""
    if xml_text is None:
        return mujoco.MjModel.from_xml_path(str(xml_path))
    # A unique name per call: parallel workers load different texts from the same folder.
    tmp = xml_path.with_name(f".{xml_path.stem}_{uuid.uuid4().hex}.xml")
    tmp.write_text(xml_text)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()


def replay(m: mujoco.MjModel, jm: JointMap, u_deg: np.ndarray, y0_deg: np.ndarray, dt: float,
           ctrl_shift_s: float = 0.0, ctrl_offset_deg: np.ndarray | None = None, gravity: bool = True) -> np.ndarray:
    """Joint angles (deg) at every logged frame [N, J] when the logged commands drive the model open loop.

    Commands are interpolated linearly between frames (as in servo.simulate) and optionally shifted later by
    ctrl_shift_s (a delay outside the model) and offset per joint. The arm starts at the first measured pose, at
    rest, with any actuator delay buffer pre-filled with the first command.
    """
    m = copy.copy(m)  # gravity is switched off on a copy, never on the caller's model
    if not gravity:
        m.opt.gravity[:] = 0.0
    d = mujoco.MjData(m)
    act = [m.actuator(n).id for n in jm.names]
    jnt = [m.joint(n).qposadr[0] for n in jm.names]
    u = u_deg + (0.0 if ctrl_offset_deg is None else ctrl_offset_deg)
    n = len(u)
    ts = m.opt.timestep
    q0 = jm.to_model(y0_deg)

    def ctrl_at(t: float) -> np.ndarray:
        x = min(max((t - ctrl_shift_s) / dt, 0.0), n - 1.0)
        i = min(int(x), n - 2)
        f = x - i
        return jm.to_model(u[i] * (1 - f) + u[i + 1] * f)

    # Pre-fill the delay history: hold the start pose while the buffer fills with the first command.
    prefill = math.ceil(float(np.max(m.actuator_delay, initial=0.0)) / ts) + 2
    for _ in range(prefill):
        d.qpos[jnt] = q0
        d.qvel[:] = 0.0
        d.ctrl[act] = ctrl_at(0.0)
        mujoco.mj_step(m, d)
    d.qpos[jnt] = q0
    d.qvel[:] = 0.0
    t0 = d.time
    out = np.empty((n, len(jnt)))
    steps_per_frame = dt / ts
    k = 0
    for i in range(n):
        target = round(i * steps_per_frame)
        while k < target:
            d.ctrl[act] = ctrl_at(d.time - t0)
            mujoco.mj_step(m, d)
            k += 1
        out[i] = d.qpos[jnt]
    return jm.to_log(out)


def effective_inertia(m: mujoco.MjModel, jm: JointMap, pose_deg: np.ndarray) -> np.ndarray:
    """Diagonal of the joint-space mass matrix (armature included) at a pose: the inertia each servo drives."""
    d = mujoco.MjData(m)
    jnt = [m.joint(n).qposadr[0] for n in jm.names]
    dof = [m.joint(n).dofadr[0] for n in jm.names]
    d.qpos[jnt] = jm.to_model(pose_deg)
    mujoco.mj_forward(m, d)
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    return full[dof, dof]


def servo_gains(law_id: str, p: dict[str, float], inertia: float) -> dict[str, float]:
    """kp, kv (N m per rad, N m s per rad), delay (s) and frictionloss (N m) that make a MuJoCo position actuator
    on a joint of this inertia follow the identified law.

    A first-order lag tau has no inertia; it is written as an overdamped pair with poles at 1/tau and 8/tau, whose
    step response differs from the lag by about tau/8. A deadband db becomes dry friction of kp * db, which gives the
    same band of commands that move nothing. The angle-dependent offset is not expressible in MJCF and is applied to
    the command instead (command_correction()).
    """
    law = LAWS[law_id]
    if law.base == "so2":
        wn, zeta = p["wn"], p["zeta"]
    else:
        p1, p2 = 1.0 / p["tau"], 8.0 / p["tau"]
        wn, zeta = math.sqrt(p1 * p2), (p1 + p2) / (2.0 * math.sqrt(p1 * p2))
    kp = wn * wn * inertia
    kv = 2.0 * zeta * wn * inertia
    fl = kp * math.radians(p.get("db", 0.0)) if "deadband" in law.features else 0.0
    return {"kp": kp, "kv": kv, "delay": p["d"], "frictionloss": fl}


def command_correction(law_id: str, p: dict[str, float], u_deg: np.ndarray) -> np.ndarray:
    """Per-frame command offset (deg) that carries b and the angle-dependent offset, evaluated at the command
    (the servo sits close to its command, so sag(theta) ~ sag(u))."""
    off = np.full_like(u_deg, p["b"], dtype=float)
    if "sag" in LAWS[law_id].features:
        r = np.deg2rad(u_deg)
        off = off + p["sc"] * np.cos(r) + p["ss"] * np.sin(r)
    return off


def write_identified(xml: str, gains: dict[str, dict[str, float]], note: str) -> str:
    """The original MJCF text with each named actuator given explicit kp, kv, delay and each joint frictionloss.
    Text edits, not a re-serialisation, so the diff against the original shows only what changed."""
    out = xml
    for name, g in gains.items():
        pat = re.compile(rf'<position class="(\w+)" name="{name}" joint="{name}" inheritrange="1"/>')
        if not pat.search(out):
            raise ValueError(f"actuator {name} not found in the MJCF; the source file changed")
        nsample = max(2, math.ceil(g["delay"] / 0.002) + 2)  # history long enough at the 0.002 s timestep
        out = pat.sub(
            rf'<position class="\1" name="{name}" joint="{name}" inheritrange="1" kp="{g["kp"]:.4g}" '
            rf'kv="{g["kv"]:.4g}" delay="{g["delay"]:.4f}" nsample="{nsample}" interp="linear"/>', out)
        jpat = re.compile(rf'<joint name="{name}" class="(\w+)"/>')
        if not jpat.search(out):
            raise ValueError(f"joint {name} not found in the MJCF; the source file changed")
        out = jpat.sub(rf'<joint name="{name}" class="\1" frictionloss="{g["frictionloss"]:.4g}"/>', out)
    return out.replace('<mujoco model="so_arm100">', f'<mujoco model="so_arm100">\n  <!-- {note} -->', 1)
