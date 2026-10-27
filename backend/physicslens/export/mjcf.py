"""MuJoCo MJCF export of a fitted system, plus a check that MuJoCo reproduces the fitted motion.

Frames: MuJoCo is z-up; the measured plane maps to x (right) and z (up). Mass never appears in these
equations of motion, so bodies get a nominal 1 kg and damping/stiffness are scaled to match.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ..physics.coords import Coordinates
from ..physics.fit import simulate
from ..physics.models import MODELS

NOMINAL_MASS = 1.0
TIMESTEP = 0.0005


class MjcfError(ValueError):
    pass


def _header(name: str, gravity: float, note: str) -> str:
    return (f'<mujoco model="{name}">\n'
            f"  <!-- Exported by PhysicsLens from a phone video. {note} -->\n"
            f'  <option timestep="{TIMESTEP}" integrator="RK4" gravity="0 0 {-gravity:.6g}"/>\n'
            '  <visual><global offwidth="640" offheight="480"/></visual>\n'
            "  <worldbody>\n"
            '    <light pos="0 -2 2" dir="0 1 -1"/>\n')


def build_mjcf(fit_state: dict[str, Any]) -> tuple[str, list[str]]:
    """(xml, notes). Notes list anything the MJCF can't express exactly, so nobody is misled."""
    spec = MODELS[fit_state["model_id"]]
    p, ctx = fit_state["params"], fit_state["ctx"]
    notes: list[str] = []
    if spec.family == "pendulum":
        L = ctx["length_m"]
        inertia = NOMINAL_MASS * L * L  # point mass on a massless rod
        gamma = p.get("gamma", 0.0)
        if "beta" in p:
            # Quadratic drag has no joint equivalent; use the linear damping that removes the same energy per
            # cycle at the observed peak angular speed (8 beta Omega / 3 pi).
            t = np.arange(0, 2 * np.pi * np.sqrt(L / p["g"]), 0.01)
            q = simulate(spec, p, np.array(fit_state["y0"], dtype=float), t, ctx)
            peak = float(np.abs(np.gradient(q[:, 0], t)).max()) if q is not None else 0.0
            gamma = 8 * p["beta"] * peak / (3 * np.pi)
            notes.append(f"Air drag (beta w|w|) has no MuJoCo joint equivalent; exported as the energy-equivalent "
                         f"linear damping at the observed swing speed (gamma = {gamma:.4g}/s). Use the Python script for the exact model.")
        damping = gamma * inertia
        if spec.id == "pendulum_small_angle":
            notes.append("MuJoCo simulates the full nonlinear pendulum; the small-angle form is its limit for small swings.")
        xml = (_header("physicslens_pendulum", p["g"], f"Pendulum, L = {L:.4g} m, g = {p['g']:.4g} m/s^2.")
               + '    <body name="pivot" pos="0 0 0">\n'
               # axis -y makes a positive angle swing the bob toward +x, matching the measured angle.
               + f'      <joint name="swing" type="hinge" axis="0 -1 0" damping="{damping:.6g}"/>\n'
               + f'      <inertial pos="0 0 {-L:.6g}" mass="{NOMINAL_MASS}" diaginertia="1e-9 1e-9 1e-9"/>\n'
               + f'      <geom type="capsule" fromto="0 0 0 0 0 {-L:.6g}" size="0.004" rgba="0.8 0.8 0.8 1" mass="0"/>\n'
               + f'      <geom type="sphere" pos="0 0 {-L:.6g}" size="{max(0.01, 0.06 * L):.4g}" rgba="0.13 0.83 0.93 1" mass="0"/>\n'
               + "    </body>\n  </worldbody>\n</mujoco>\n")
    elif spec.family == "oscillator":
        omega, zeta, s0 = p["omega"], p.get("zeta", 0.0), p["s0"]
        ax, az = ctx["axis_x"], ctx["axis_y"]
        xml = (_header("physicslens_oscillator", 0.0,
                       f"Spring-mass, omega = {omega:.4g} rad/s, zeta = {zeta:.4g}. Gravity is folded into the equilibrium s0.")
               + '    <body name="mass" pos="0 0 0">\n'
               + f'      <joint name="stretch" type="slide" axis="{ax:.6g} 0 {az:.6g}" stiffness="{omega**2 * NOMINAL_MASS:.6g}"'
               + f' damping="{2 * zeta * omega * NOMINAL_MASS:.6g}" springref="{s0:.6g}"/>\n'
               + f'      <inertial pos="0 0 0" mass="{NOMINAL_MASS}" diaginertia="1e-6 1e-6 1e-6"/>\n'
               + '      <geom type="sphere" size="0.02" rgba="0.13 0.83 0.93 1" mass="0"/>\n'
               + "    </body>\n  </worldbody>\n</mujoco>\n")
    elif spec.family == "rotation":
        c, k, q = p.get("c", 0.0), p.get("k", 0.0), p.get("q", 0.0)
        if q:
            notes.append("Air drag on the spin (q w|w|) has no MuJoCo joint equivalent; exported without it. "
                         "Use the Python script for the exact model.")
        arm = ctx["arm_m"]
        # Unit moment of inertia, so joint damping equals k and dry-friction torque (frictionloss) equals c.
        xml = (_header("physicslens_spin", 0.0, f"Spin-down: dry friction c = {c:.4g} rad/s^2, viscous k = {k:.4g} 1/s.")
               + '    <body name="spinner" pos="0 0 0">\n'
               + f'      <joint name="spin" type="hinge" axis="0 -1 0" damping="{k:.6g}" frictionloss="{c:.6g}"/>\n'
               + f'      <inertial pos="0 0 0" mass="{NOMINAL_MASS}" diaginertia="1 1 1"/>\n'
               + f'      <geom type="capsule" fromto="{-arm:.6g} 0 0 {arm:.6g} 0 0" size="{max(0.005, arm / 8):.4g}" rgba="0.13 0.83 0.93 1" mass="0"/>\n'
               + "    </body>\n  </worldbody>\n</mujoco>\n")
    elif spec.family == "projectile":
        if spec.id != "free_fall":
            notes.append("Drag is not representable without object geometry for MuJoCo's fluid model; exported "
                         "without drag. Use the Python script for the exact model.")
        xml = (_header("physicslens_projectile", p["g"], f"Projectile, g = {p['g']:.4g} m/s^2.")
               + '    <body name="object" pos="0 0 0">\n      <freejoint/>\n'
               + f'      <inertial pos="0 0 0" mass="{NOMINAL_MASS}" diaginertia="1e-6 1e-6 1e-6"/>\n'
               + '      <geom type="sphere" size="0.02" rgba="0.13 0.83 0.93 1" mass="0" contype="0" conaffinity="0"/>\n'
               + "    </body>\n  </worldbody>\n</mujoco>\n")
    else:
        raise MjcfError(f"No MJCF export for family {spec.family!r}.")
    return xml, notes


def simulate_mjcf(xml: str, fit_state: dict[str, Any], seconds: float) -> np.ndarray:
    """Generalized coordinate(s) over time at the clip's frame rate, starting from the measured state."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    family = MODELS[fit_state["model_id"]].family
    y0 = np.array(fit_state["y0"], dtype=float)
    d = len(y0) // 2
    if family == "projectile":
        data.qpos[:3] = [y0[0], 0.0, y0[1]]
        data.qvel[:3] = [y0[2], 0.0, y0[3]]
    else:
        data.qpos[0], data.qvel[0] = y0[0], y0[1]
    mujoco.mj_forward(model, data)
    fps = fit_state["fps"]
    n = int(seconds * fps)
    # A step that divides the frame interval exactly; rounding (e.g. 67 x 0.5 ms for 30 fps) drifts 0.5% per frame.
    steps_per_frame = max(1, int(np.ceil(1.0 / (fps * model.opt.timestep))))
    model.opt.timestep = 1.0 / (fps * steps_per_frame)
    out = np.zeros((n, d))
    for i in range(n):
        out[i] = [data.qpos[0], data.qpos[2]] if family == "projectile" else [data.qpos[0]]
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
    return out


def check_against_fit(fit_state: dict[str, Any], seconds: float) -> dict[str, Any]:
    """Run the exported MJCF in MuJoCo and compare with the fitted ODE; ghost points for the video."""
    spec = MODELS[fit_state["model_id"]]
    xml, notes = build_mjcf(fit_state)
    q_mj = simulate_mjcf(xml, fit_state, seconds)
    t = np.arange(len(q_mj)) / fit_state["fps"]
    q_fit = simulate(spec, fit_state["params"], np.array(fit_state["y0"]), t, fit_state["ctx"])
    coords = Coordinates(spec.family, t, q_mj, (), fit_state["ctx"])
    xy_mj = coords.to_xy(q_mj)
    rms_m = None
    if q_fit is not None:
        xy_fit = Coordinates(spec.family, t, q_fit, (), fit_state["ctx"]).to_xy(q_fit)
        rms_m = float(np.sqrt(np.mean(np.sum((xy_mj - xy_fit) ** 2, axis=1))))
    ox, oy = fit_state["origin_px"]
    ppm, f0 = fit_state["px_per_m"], int(fit_state["first_frame"])
    points = [[f0 + i, round(ox + x * ppm, 2), round(oy - y * ppm, 2)] for i, (x, y) in enumerate(xy_mj)]
    return {"rms_vs_fit_m": rms_m, "notes": notes, "points": points}
