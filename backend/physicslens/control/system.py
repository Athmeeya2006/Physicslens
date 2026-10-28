"""The controlled system for a task: which goals each family supports, the robustness set, the numbers a
controller is given, the textbook baseline controller and the simple bounds that prove a goal impossible.

Everything here is our own code and local math; the generated controller only ever sees the `params` dict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..agent.schemas import ControlTask
from ..export.mjcf import NOMINAL_MASS, build_mjcf
from ..export.sim2real import (
    CONTROL_DT,
    EARTH_G,
    GAIN_NORM,
    START_ANGLE,
    actuated_mjcf,
    bootstrap_plants,
    measured_coordinates,
)
from ..physics.custom import spec_for
from ..physics.fit import FitResult, simulate
from ..physics.uncertainty import bootstrap_samples

HOLD_S = 1.0  # after settling, the error must stay inside the tolerance at least this long
SUPPORTED: dict[str, tuple[str, ...]] = {
    "pendulum": ("reach_and_hold", "swing_up", "stop", "track_sine"),
    "oscillator": ("reach_and_hold", "stop", "track_sine"),
}
PENDULUM_KEYS = ("g", "gamma", "beta")
GOAL_TEXT = {
    "reach_and_hold": "move to the target and hold it there",
    "swing_up": "swing up from hanging to upright (angle pi) and balance there",
    "stop": "bring the motion to rest",
    "track_sine": "follow a sine reference",
}


class ControlUnsupported(ValueError):
    """An honest "not supported yet" (or a task that does not fit this result); the message is shown as is."""


@dataclass(frozen=True)
class Plant:
    """One variant of the identified system: parameters plus the MuJoCo model built from them."""

    params: dict[str, float]
    xml: str
    q0: float
    qdot0: float


def check_supported(result: dict[str, Any], task: ControlTask) -> None:
    if result.get("status") != "accepted" or "fit_state" not in result:
        raise ControlUnsupported("Controllers need an accepted model of the system first.")
    family = spec_for(result["fit_state"]).family
    if task.family != family:
        raise ControlUnsupported(f"This analysis identified a {family}, not a {task.family}.")
    if family not in SUPPORTED:
        raise ControlUnsupported(f"Controller tasks for a {family} are not supported yet.")
    if task.goal not in SUPPORTED[family]:
        raise ControlUnsupported(f"The goal {task.goal.replace('_', ' ')} is not supported yet for a {family}.")
    if family == "oscillator":
        _, notes = build_mjcf(result["fit_state"])
        if notes:
            raise ControlUnsupported("This oscillator's law has no exact MuJoCo form, so controllers can't be tested on it yet.")


def _sample_params(result: dict[str, Any], reps: int, seed: int) -> list[dict[str, float]]:
    """Bootstrap refits of the accepted model: the same spread its uncertainty intervals come from."""
    fs = result["fit_state"]
    spec = spec_for(fs)
    c = measured_coordinates(result)
    if spec.family == "pendulum":
        return bootstrap_plants(spec, c, fs, reps, seed=seed)
    y0 = np.array(fs["y0"], dtype=float)
    sim = simulate(spec, fs["params"], y0, c.t, c.ctx)
    if sim is None:
        raise ControlUnsupported("The accepted model could not be re-simulated.")
    fit = FitResult(spec.id, dict(fs["params"]), y0, sim.size, len(spec.params) + len(y0), 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, success=True, sim=sim)
    samples, _ = bootstrap_samples(spec, c, fit, reps=reps, seed=seed)
    return [dict(zip(spec.param_names, (float(v) for v in row), strict=True)) for row in samples]


def robustness_params(result: dict[str, Any], reps: int, seed: int) -> list[dict[str, float]]:
    """Nominal identified parameters first, then `reps` bootstrap samples."""
    fs = result["fit_state"]
    if spec_for(fs).family == "pendulum":
        p = fs["params"]
        nominal = {"g": float(p["g"]), "gamma": float(p.get("gamma", 0.0)), "beta": float(p.get("beta", 0.0)),
                   "L": float(fs["ctx"]["length_m"])}
    else:
        nominal = {k: float(v) for k, v in fs["params"].items()}
    return [nominal, *_sample_params(result, reps, seed)]


def _start(fs: dict[str, Any], task: ControlTask, sample: dict[str, float]) -> tuple[float, float]:
    family = spec_for(fs).family
    y0 = [float(v) for v in fs["y0"]]
    if task.goal == "stop":
        return y0[0], y0[1]  # the motion seen in the clip
    if family == "pendulum":
        return (START_ANGLE if task.goal == "swing_up" else 0.0), 0.0
    return sample["s0"], 0.0  # at rest


def build_plants(fs: dict[str, Any], task: ControlTask, samples: list[dict[str, float]]) -> list[Plant]:
    family = spec_for(fs).family
    plants = []
    for s in samples:
        if family == "pendulum":
            # Only the accepted model's own terms, so the MJCF builder sees the same law with sampled values.
            params = {k: s[k] for k in PENDULUM_KEYS if k in fs["params"]}
            variant = {**fs, "params": params, "ctx": {**fs["ctx"], "length_m": s["L"]}}
            xml = actuated_mjcf(variant, task.actuator_limit)
        else:
            variant = {**fs, "params": dict(s)}
            xml = actuated_mjcf(variant, task.actuator_limit).replace('joint="swing"', 'joint="stretch"')
        q0, qdot0 = _start(fs, task, s)
        plants.append(Plant(s, xml, q0, qdot0))
    return plants


def controller_params(fs: dict[str, Any], task: ControlTask) -> dict[str, float]:
    """The numbers the controller gets: the nominal identified model and the task, nothing about the samples."""
    p = fs["params"]
    out: dict[str, float] = {"u_max": task.actuator_limit, "mass": NOMINAL_MASS, "tolerance": task.tolerance,
                             "time_limit_s": task.time_limit_s, "frequency_hz": task.frequency_hz}
    if spec_for(fs).family == "pendulum":
        L = float(fs["ctx"]["length_m"])
        out.update(g=float(p["g"]), gamma=float(p.get("gamma", 0.0)), beta=float(p.get("beta", 0.0)), L=L,
                   inertia=NOMINAL_MASS * L * L, target=math.pi if task.goal == "swing_up" else task.target)
    else:
        out.update({k: float(v) for k, v in p.items()})
        out.setdefault("zeta", 0.0)
        out["target"] = task.target
    return out


def error_spec(fs: dict[str, Any], task: ControlTask) -> dict[str, Any]:
    """How the harness measures the error each step (computed by the harness, never by the controller)."""
    pendulum = spec_for(fs).family == "pendulum"
    if task.goal == "stop":
        return {"kind": "velocity"}
    offset = 0.0 if pendulum else float(fs["params"]["s0"])
    target = math.pi if task.goal == "swing_up" else task.target
    return {"kind": "angle" if pendulum else "position", "offset": offset, "target": target,
            "sine": task.goal == "track_sine", "amplitude": task.target, "frequency_hz": task.frequency_hz}


def evaluation_job(fs: dict[str, Any], task: ControlTask, plants: list[Plant]) -> dict[str, Any]:
    return {
        "params": controller_params(fs, task), "dt": CONTROL_DT, "horizon_s": task.time_limit_s + HOLD_S,
        "hold_s": HOLD_S, "time_limit_s": task.time_limit_s, "u_max": task.actuator_limit,
        "tolerance": task.tolerance, "error": error_spec(fs, task),
        "samples": [{"xml": p.xml, "q0": p.q0, "qdot0": p.qdot0} for p in plants],
    }


def infeasibility(fs: dict[str, Any], task: ControlTask) -> str | None:
    """A note when a simple force or torque bound on the nominal model shows the goal cannot be met at all."""
    p = controller_params(fs, task)
    limit = task.actuator_limit
    w = 2 * math.pi * task.frequency_hz
    if spec_for(fs).family == "pendulum":
        if task.goal == "reach_and_hold":
            need = NOMINAL_MASS * p["g"] * p["L"] * abs(math.sin(task.target))
            if need > limit:
                return (f"Holding the pendulum at {math.degrees(task.target):.0f} degrees takes a steady torque of "
                        f"m g L sin(target) = {need:.3g} N m per kg, more than the motor's {limit:.3g}.")
        if task.goal == "track_sine" and abs(task.target) <= 0.5:
            # Small swings are close to linear, where the torque a sine needs is known exactly.
            need = p["inertia"] * abs(task.target) * math.hypot(p["g"] / p["L"] - w * w, p["gamma"] * w)
            if need > limit:
                return (f"Following a {math.degrees(abs(task.target)):.0f} degree sine at {task.frequency_hz:.3g} Hz takes "
                        f"about {need:.3g} N m per kg of torque, more than the motor's {limit:.3g}.")
        return None
    omega, zeta = p["omega"], p["zeta"]
    if task.goal == "reach_and_hold":
        need = NOMINAL_MASS * omega**2 * abs(task.target)
        if need > limit:
            return (f"Holding the mass {abs(task.target) * 100:.3g} cm from rest takes a steady force of "
                    f"m omega^2 x = {need:.3g} N per kg, more than the actuator's {limit:.3g}.")
    if task.goal == "track_sine":
        need = NOMINAL_MASS * abs(task.target) * math.hypot(omega**2 - w * w, 2 * zeta * omega * w)
        if need > limit:
            return (f"Following that sine takes a force of {need:.3g} N per kg at its peak, more than the "
                    f"actuator's {limit:.3g}.")
    return None


# ---------- the textbook baseline, written as a controller under the same contract ----------

_BASELINE = '''"""Textbook controller: tuned on the textbook model (no damping or drag, {g_text})."""
import math

G = {g!r}
L = {L!r}
OMEGA = {omega!r}
WN = {wn!r}
GAIN = {gain!r}
GOAL = {goal!r}


def control(t, q, qdot, params):
    th, w = q[0], qdot[0]
    m = params["mass"]
    if GOAL == "stop":
        return [-m * {inertia_expr} * 2.0 * WN * w]
    if GOAL == "swing_up":
        err = (th - math.pi + math.pi) % (2.0 * math.pi) - math.pi
        inertia = m * L * L
        if abs(err) < 0.35:
            return [m * G * L * math.sin(th) - inertia * (WN * WN * err + 2.0 * WN * w)]
        energy = 0.5 * inertia * w * w + m * G * L * (1.0 - math.cos(th))
        return [GAIN * (2.0 * m * G * L - energy) * w]
    ref, ref_v, ref_a = params["target"], 0.0, 0.0
    if GOAL == "track_sine":
        k = 2.0 * math.pi * params["frequency_hz"]
        ref = params["target"] * math.sin(k * t)
        ref_v = params["target"] * k * math.cos(k * t)
        ref_a = -params["target"] * k * k * math.sin(k * t)
    {body}
'''

_PENDULUM_BODY = '''err = (th - ref + math.pi) % (2.0 * math.pi) - math.pi
    inertia = m * L * L
    return [m * G * L * math.sin(th) + inertia * (ref_a - WN * WN * err - 2.0 * WN * (w - ref_v))]'''

_OSCILLATOR_BODY = '''x = th - params["s0"]
    err = x - ref
    return [m * (OMEGA * OMEGA * x + ref_a - WN * WN * err - 2.0 * WN * (w - ref_v))]'''


def textbook_controller(fs: dict[str, Any], task: ControlTask, calibrated: bool) -> str:
    """Computed-torque PD (energy pumping then PD for a swing-up), every constant from the textbook model.

    The same law a textbook would give, tuned on the same model the sim-to-real panel calls "textbook": Earth's g,
    no loss terms. Its gains put the settling time at about half the time limit.
    """
    pendulum = spec_for(fs).family == "pendulum"
    wn = 8.0 / task.time_limit_s
    L = float(fs["ctx"]["length_m"]) if pendulum else 0.0
    g = (EARTH_G if calibrated else float(fs["params"]["g"])) if pendulum else 0.0
    gain = 0.0
    if pendulum:
        # sim2real's energy-shaping normalization, aimed at the upright energy instead of a 60 degree swing.
        energy = 2 * NOMINAL_MASS * g * L
        gain = GAIN_NORM * task.actuator_limit / (energy * math.sqrt(2 * energy / (NOMINAL_MASS * L * L)))
    return _BASELINE.format(
        g_text=("g = 9.81 m/s^2" if calibrated else "g as fitted") if pendulum else "the measured frequency",
        g=g, L=L, omega=float(fs["params"].get("omega", 0.0)), wn=wn, gain=gain, goal=task.goal,
        inertia_expr="L * L" if pendulum else "1.0", body=_PENDULUM_BODY if pendulum else _OSCILLATOR_BODY)
