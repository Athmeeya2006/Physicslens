"""The sim-to-real gap, made visible.

Part 1 is evidence against the real video: the textbook version of the system (no damping, drag or friction,
g = 9.81 m/s^2) and the PhysicsLens model are both simulated and compared with the measured motion, on the
held-out last quarter and on the whole clip. Neither model sees the held-out quarter: the PhysicsLens
parameters are refitted on the first three quarters for this comparison.

Part 2 is simulation only, for pendulums: a torque-limited motor at the pivot (the exported MJCF plus an
actuator) pumps the swing to a target amplitude. One controller is tuned on the textbook model, one on the
PhysicsLens model, and both are run in MuJoCo on pendulums drawn from the bootstrap refits of the PhysicsLens
model. The "real system" there is the verified model, not the video.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

from ..physics.coords import Coordinates, to_coordinates
from ..physics.custom import spec_for
from ..physics.fit import TRAIN_FRACTION, FitResult, fit_window, simulate
from ..physics.models import MODELS, ModelSpec
from ..physics.uncertainty import bootstrap_samples
from ..sigproc.cleanup import differentiate
from ..sigproc.trajectory import Trajectory
from .mjcf import NOMINAL_MASS, build_mjcf

EARTH_G = 9.81
PLOT_POINTS = 400
SIM_LABEL = "Simulation. The 'real system' here is the verified model; Part 1 is the evidence against real video."

# Part 2: pump a pendulum from nearly at rest to a 60 degree swing with a motor too weak to lift it directly.
TARGET_AMPLITUDE = math.radians(60.0)
TORQUE_FRACTION = 0.3  # of m g L, the torque that holds the pendulum horizontal
GAIN_NORM = 4.0  # energy gain, in units of (torque limit) / (target energy x target peak speed)
START_ANGLE = 0.05  # rad, released at rest
HORIZON_S = 12.0
CONTROL_DT = 0.002  # s: MuJoCo step and control rate (500 Hz)
SETTLE_BAND = 0.02  # amplitude within 2% of target
FINAL_WINDOW_S = 2.0


class Sim2RealError(ValueError):
    pass


# ---------- Part 1: textbook and PhysicsLens models against the measured motion ----------


def measured_coordinates(result: dict[str, Any]) -> Coordinates:
    """The measured generalized coordinate, rebuilt from the stored pixel track with the fit's own geometry."""
    fs = result["fit_state"]
    spec = spec_for(fs)
    pts = np.array(result["real_track_px"], dtype=float)
    if len(pts) < 10:
        raise Sim2RealError("This analysis has too little track to compare.")
    frames = pts[:, 0].astype(int)
    ox, oy = fs["origin_px"]
    ppm = float(fs["px_per_m"])
    # Same frame as the fit: meters, y up, origin at the track's first point (or the image origin for a spin).
    xy = np.column_stack([(pts[:, 1] - ox) / ppm, (oy - pts[:, 2]) / ppm])
    traj = Trajectory(t=(frames - frames[0]) / float(fs["fps"]), xy=xy, frames=frames, fps=float(fs["fps"]))
    ctx = {k: float(v) for k, v in fs["ctx"].items()}
    if spec.family == "rotation":
        ph = np.unwrap(np.arctan2(xy[:, 1] - ctx["center_y"], xy[:, 0] - ctx["center_x"]))
        # The rim marker fixes the angle only up to whole turns; a torsion spring's rest angle needs the same turn.
        ph += 2 * np.pi * round((float(fs["y0"][0]) - ph[0]) / (2 * np.pi))
        return Coordinates("rotation", traj.t, ph[:, None], ("phi",), ctx)
    if spec.family in ("pendulum", "oscillator", "line"):
        return to_coordinates(traj, spec.family, geometry=ctx)
    if spec.family in ("projectile", "plane"):
        return Coordinates(spec.family, traj.t, xy.copy(), ("x", "y"), ctx)
    raise Sim2RealError(f"No sim-to-real comparison for a {spec.family} model.")


def textbook_model(spec: ModelSpec, fitted: dict[str, float],
                   calibrated: bool) -> tuple[ModelSpec, dict[str, float], tuple[str, ...], str]:
    """The same system as a textbook would write it: (model, parameters, parameters to fit, description).

    Every loss term is removed and g is Earth's. Constants a textbook can't know (a spring's stiffness, its rest
    position) keep their fitted values, so the comparison isolates what the textbook leaves out. Without a scale
    9.81 m/s^2 means nothing in the clip's units, so g is then fitted on the training window like the rest.
    """
    g_free: tuple[str, ...] = () if calibrated else ("g",)
    g_text = "g = 9.81 m/s²" if calibrated else "g fitted (no scale in view to set 9.81 m/s²)"
    g = EARTH_G if calibrated else fitted.get("g", EARTH_G)
    fam = spec.family
    if fam == "pendulum":
        return MODELS["pendulum"], {"g": g}, g_free, f"Frictionless pendulum, {g_text}, same length"
    if fam == "oscillator":
        return (MODELS["harmonic"], {"omega": fitted["omega"], "s0": fitted["s0"]}, (),
                "Undamped spring at the measured frequency, no friction")
    if fam == "projectile":
        return MODELS["free_fall"], {"g": g}, g_free, f"Free fall without air drag, {g_text}"
    if fam == "rotation":
        if "kappa" in fitted:
            return (MODELS["spin_spring"], {"kappa": fitted["kappa"], "phi0": fitted["phi0"], "k": 0.0}, (),
                    "Undamped torsional spring at the measured stiffness")
        return MODELS["spin_free"], {}, (), "Frictionless spin"
    if fam == "line":
        return MODELS["line_constant_velocity"], {}, (), "Nothing acts on it (constant velocity)"
    if fam == "plane":
        return (MODELS["plane_constant_accel"], {"ax0": 0.0, "ay0": -g}, () if calibrated else ("ay0",),
                f"Free fall without air drag, {g_text}")
    raise Sim2RealError(f"No textbook model for a {fam} system.")


def _state_bounds(c: Coordinates) -> tuple[np.ndarray, np.ndarray]:
    """The same initial-state box the main fit searches."""
    d = c.q.shape[1]
    vel = differentiate(c.q, 1.0 / (c.t[1] - c.t[0])).vel
    span = np.ptp(c.q, axis=0).max() + 1e-9
    vspan = np.abs(vel).max() + 1e-9
    return (np.concatenate([c.q.min(axis=0) - span, -3 * vspan * np.ones(d)]),
            np.concatenate([c.q.max(axis=0) + span, 3 * vspan * np.ones(d)]))


def _fit_textbook(spec: ModelSpec, params: dict[str, float], free: tuple[str, ...], c: Coordinates, stop: int,
                  y0: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Best starting state (and any free parameter) on the training window: the textbook model's fairest chance."""
    lo, hi = _state_bounds(c)
    meta = {p.name: p for p in spec.params}
    # ay0 has no Param bounds of its own worth trusting here; a g-like range in the clip's units is enough.
    p_lo = [meta[k].lower if k in meta and k != "ay0" else -1e3 for k in free]
    p_hi = [meta[k].upper if k in meta and k != "ay0" else 1e3 for k in free]
    lower, upper = np.concatenate([p_lo, lo]), np.concatenate([p_hi, hi])
    t, q = c.t[:stop], c.q[:stop]

    def unpack(x: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        return {**params, **dict(zip(free, (float(v) for v in x[: len(free)]), strict=True))}, x[len(free):]

    def residual(x: np.ndarray) -> np.ndarray:
        p, s0 = unpack(x)
        sim = simulate(spec, p, s0, t, c.ctx)
        return np.full(q.size, 1e3) if sim is None else (sim - q).ravel()

    x0 = np.concatenate([[params[k] for k in free], y0])
    return unpack(least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper), x_scale="jac", max_nfev=200).x)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


PLOT_LABEL = {"pendulum": ("Swing angle", "rad"), "oscillator": ("Displacement", "m"), "projectile": ("Height", "m"),
              "rotation": ("Spin angle", "rad"), "line": ("Position along the path", "m"), "plane": ("Height", "m")}


def compare_to_video(c: Coordinates, fit_state: dict[str, Any], calibrated: bool) -> dict[str, Any]:
    """Textbook against PhysicsLens against the measured motion; both models fitted on the first 75% only."""
    spec = spec_for(fit_state)
    n = len(c.t)
    n_train = int(n * TRAIN_FRACTION)
    if n - n_train < 3:
        raise Sim2RealError("The clip is too short to hold out a quarter of it.")
    # PhysicsLens: refit from the accepted values, but on the training window only, so the held-out quarter is unseen.
    lo, hi = _state_bounds(c)
    theta0 = np.concatenate([[fit_state["params"][p] for p in spec.param_names], np.array(fit_state["y0"], dtype=float)])
    lower = np.concatenate([[p.lower for p in spec.params], lo])
    upper = np.concatenate([[p.upper for p in spec.params], hi])
    theta = fit_window(spec, c, n_train, theta0, lower, upper)
    k = len(spec.params)
    fitted = dict(zip(spec.param_names, (float(v) for v in theta[:k]), strict=True))
    y0_fit = theta[k:]
    sim_fit = simulate(spec, fitted, y0_fit, c.t, c.ctx)

    tb_spec, tb_params, tb_free, tb_text = textbook_model(spec, fitted, calibrated)
    tb_params, y0_tb = _fit_textbook(tb_spec, tb_params, tb_free, c, n_train, y0_fit)
    sim_tb = simulate(tb_spec, tb_params, y0_tb, c.t, c.ctx)
    if sim_fit is None or sim_tb is None:
        raise Sim2RealError("A simulation diverged; no comparison for this clip.")

    def errors(sim: np.ndarray) -> dict[str, float | None]:
        out: dict[str, float | None] = {"heldout": _rms(sim[n_train:], c.q[n_train:]), "whole": _rms(sim, c.q)}
        if calibrated and c.family != "rotation":
            # Distance in the image plane, in meters: the same yardstick for every family.
            real = c.to_xy(c.q)
            out["heldout_m"] = _rms(c.to_xy(sim)[n_train:], real[n_train:])
            out["whole_m"] = _rms(c.to_xy(sim), real)
        return out

    e_tb, e_fit = errors(sim_tb), errors(sim_fit)
    label, unit = PLOT_LABEL.get(c.family, (c.names[0], ""))
    col = 1 if c.family in ("projectile", "plane") else 0
    idx = np.arange(0, n, max(1, n // PLOT_POINTS))
    return {
        "family": c.family, "label": label, "unit": unit, "calibrated": calibrated,
        "heldout_start": round(float(c.t[n_train]), 4),
        "textbook": {"model_id": tb_spec.id, "title": tb_spec.title, "description": tb_text,
                     "params": tb_params, "errors": e_tb},
        "physicslens": {"model_id": spec.id, "title": spec.title, "params": fitted, "errors": e_fit},
        # How many times further the textbook model lands from the real motion than the PhysicsLens model.
        "ratio_heldout": e_tb["heldout"] / max(e_fit["heldout"], 1e-12),
        "ratio_whole": e_tb["whole"] / max(e_fit["whole"], 1e-12),
        "t": [round(float(v), 4) for v in c.t[idx]],
        "measured": [round(float(v), 5) for v in c.q[idx, col]],
        "textbook_series": [round(float(v), 5) for v in sim_tb[idx, col]],
        "physicslens_series": [round(float(v), 5) for v in sim_fit[idx, col]],
    }


# ---------- Part 2: a torque-limited swing-up in MuJoCo, tuned on each model ----------


def pendulum_constants(params: dict[str, float], length: float) -> dict[str, float]:
    return {"g": float(params["g"]), "gamma": float(params.get("gamma", 0.0)), "beta": float(params.get("beta", 0.0)),
            "L": float(length)}


def tune_swing_up(model: dict[str, float], torque_limit: float) -> dict[str, float]:
    """Energy-shaping swing-up, with every constant taken from the model it is tuned on.

    Torque k (E* - E) w pumps energy in while the swing is short of the target and takes it out past it; the
    model's own damping torque is added back, so the energy the model expects to lose is replaced. A model
    without damping adds nothing, and on a damped pendulum the swing then settles where loss and pumping balance.
    """
    inertia = NOMINAL_MASS * model["L"] ** 2
    energy = NOMINAL_MASS * model["g"] * model["L"] * (1 - math.cos(TARGET_AMPLITUDE))
    peak_speed = math.sqrt(2 * energy / inertia)
    return {"g": model["g"], "L": model["L"], "inertia": inertia, "target_energy": energy,
            "gain": GAIN_NORM * torque_limit / (energy * peak_speed), "gamma": model["gamma"], "beta": model["beta"]}


def swing_up_torque(ctrl: dict[str, float], th: float, w: float) -> float:
    energy = 0.5 * ctrl["inertia"] * w * w + NOMINAL_MASS * ctrl["g"] * ctrl["L"] * (1 - math.cos(th))
    loss = ctrl["inertia"] * (ctrl["gamma"] * w + ctrl["beta"] * w * abs(w))
    return ctrl["gain"] * (ctrl["target_energy"] - energy) * w + loss


def actuated_mjcf(fit_state: dict[str, Any], torque_limit: float) -> str:
    """The exported pendulum with a motor at the pivot; MuJoCo clamps the command to the torque limit."""
    xml, _ = build_mjcf(fit_state)
    motor = (f'  <actuator>\n    <motor name="pivot_motor" joint="swing" gear="1" ctrllimited="true" '
             f'ctrlrange="{-torque_limit:.6g} {torque_limit:.6g}"/>\n  </actuator>\n</mujoco>\n')
    return xml.replace("</mujoco>\n", motor)


def amplitude(th: np.ndarray, w: np.ndarray, g: float, length: float) -> np.ndarray:
    """The swing amplitude the current energy corresponds to; smooth, unlike the angle itself."""
    return np.arccos(np.clip(np.cos(th) - length * w**2 / (2 * g), -1.0, 1.0))


def run_swing_up(fit_state: dict[str, Any], ctrl: dict[str, float], torque_limit: float) -> dict[str, Any]:
    """One closed-loop run in MuJoCo; returns the amplitude over time and the step-response numbers."""
    model = mujoco.MjModel.from_xml_string(actuated_mjcf(fit_state, torque_limit))
    model.opt.timestep = CONTROL_DT
    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0] = START_ANGLE, 0.0
    mujoco.mj_forward(model, data)
    n = int(HORIZON_S / CONTROL_DT)
    th, w = np.empty(n), np.empty(n)
    for i in range(n):
        th[i], w[i] = data.qpos[0], data.qvel[0]
        data.ctrl[0] = swing_up_torque(ctrl, th[i], w[i])
        mujoco.mj_step(model, data)
    t = np.arange(n) * CONTROL_DT
    amp = amplitude(th, w, fit_state["params"]["g"], fit_state["ctx"]["length_m"])
    outside = np.abs(amp - TARGET_AMPLITUDE) > SETTLE_BAND * TARGET_AMPLITUDE
    settled = not outside[-1]
    settling = float(t[np.nonzero(outside)[0][-1] + 1]) if outside.any() and settled else (0.0 if settled else None)
    final = float(np.mean(amp[t >= HORIZON_S - FINAL_WINDOW_S]))
    return {
        "overshoot_pct": max(0.0, float(amp.max() - TARGET_AMPLITUDE) / TARGET_AMPLITUDE * 100),
        "settling_s": settling,
        "final_error_deg": math.degrees(final - TARGET_AMPLITUDE),
        "t": t, "amplitude": amp,
    }


def _stats(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def control_study(fit_state: dict[str, Any], plants: list[dict[str, float]], calibrated: bool) -> dict[str, Any]:
    """plants: pendulum constants (g, gamma, beta, L) to evaluate both controllers on, e.g. bootstrap refits."""
    p, L = fit_state["params"], float(fit_state["ctx"]["length_m"])
    identified = pendulum_constants(p, L)
    textbook = {"g": EARTH_G if calibrated else identified["g"], "gamma": 0.0, "beta": 0.0, "L": L}
    torque_limit = TORQUE_FRACTION * NOMINAL_MASS * identified["g"] * L  # one motor for both controllers
    controllers = {"textbook": tune_swing_up(textbook, torque_limit), "physicslens": tune_swing_up(identified, torque_limit)}
    runs: dict[str, list[dict[str, Any]]] = {k: [] for k in controllers}
    for plant in plants:
        # Only the accepted model's own terms, so the MJCF builder sees the same law with sampled values.
        params = {k: plant[k] for k in ("g", "gamma", "beta") if k in p}
        fs = {**fit_state, "params": params, "ctx": {**fit_state["ctx"], "length_m": plant["L"]}}
        for name, ctrl in controllers.items():
            runs[name].append(run_swing_up(fs, ctrl, torque_limit))
    summary = {}
    for name, rs in runs.items():
        settled = [r["settling_s"] for r in rs if r["settling_s"] is not None]
        summary[name] = {
            "overshoot_pct": _stats([r["overshoot_pct"] for r in rs]),
            "final_error_deg": _stats([r["final_error_deg"] for r in rs]),
            "abs_final_error_deg": _stats([abs(r["final_error_deg"]) for r in rs]),
            "settling_s": _stats(settled) if settled else None,
            "settled": len(settled), "runs": len(rs),
        }
    # One example run each, on the accepted parameters, for a plot.
    nominal = {**fit_state, "params": dict(p)}
    example = {name: run_swing_up(nominal, ctrl, torque_limit) for name, ctrl in controllers.items()}
    step = max(1, len(example["textbook"]["t"]) // PLOT_POINTS)
    return {
        "label": SIM_LABEL,
        "task": (f"Pump the swing from {math.degrees(START_ANGLE):.0f}° to {math.degrees(TARGET_AMPLITUDE):.0f}° "
                 f"and hold it there, with a pivot motor limited to {TORQUE_FRACTION:.0%} of the torque that holds the "
                 f"pendulum horizontal (energy-shaping control)."),
        "target_deg": math.degrees(TARGET_AMPLITUDE), "torque_limit_nm_per_kg": torque_limit / NOMINAL_MASS,
        "settle_band_pct": SETTLE_BAND * 100, "horizon_s": HORIZON_S,
        "tuned_on": {"textbook": {k: textbook[k] for k in ("g", "gamma", "beta")},
                     "physicslens": {k: identified[k] for k in ("g", "gamma", "beta")}},
        "plants": len(plants),
        "summary": summary,
        "example": {"t": [round(float(v), 3) for v in example["textbook"]["t"][::step]],
                    **{f"{name}_deg": [round(math.degrees(float(v)), 3) for v in r["amplitude"][::step]]
                       for name, r in example.items()}},
    }


def bootstrap_plants(spec: ModelSpec, c: Coordinates, fit_state: dict[str, Any], reps: int,
                     seed: int = 0) -> list[dict[str, float]]:
    """Pendulums drawn from the bootstrap refits of the accepted model, the same spread its intervals come from."""
    y0 = np.array(fit_state["y0"], dtype=float)
    sim = simulate(spec, fit_state["params"], y0, c.t, c.ctx)
    if sim is None:
        raise Sim2RealError("The accepted model could not be re-simulated.")
    fit = FitResult(spec.id, dict(fit_state["params"]), y0, sim.size, len(spec.params) + len(y0), 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, success=True, sim=sim)
    samples, lengths = bootstrap_samples(spec, c, fit, reps=reps, seed=seed)
    plants = []
    for row, length in zip(samples, lengths, strict=True):
        p = dict(zip(spec.param_names, (float(v) for v in row), strict=True))
        plants.append(pendulum_constants(p, c.ctx["length_m"] if np.isnan(length) else float(length)))
    return plants


def sim2real(result: dict[str, Any], reps: int = 20) -> dict[str, Any]:
    """Both parts for a stored, accepted result; the control study only for pendulums."""
    if result.get("status") != "accepted" or "fit_state" not in result:
        raise Sim2RealError("The sim-to-real comparison needs an accepted model.")
    fs = result["fit_state"]
    spec = spec_for(fs)
    calibrated = (result.get("calibration") or {}).get("source", "none") != "none"
    c = measured_coordinates(result)
    out: dict[str, Any] = {"video": compare_to_video(c, fs, calibrated)}
    if spec.family == "pendulum":
        out["control"] = control_study(fs, bootstrap_plants(spec, c, fs, reps), calibrated)
    else:
        out["control"] = None
        out["control_note"] = "The control study is for pendulums only."
    return out
