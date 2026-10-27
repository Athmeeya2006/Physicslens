"""Standalone exports: a Python simulation script (numpy + scipy only) and a JSON parameter file."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ..physics.models import MODELS

# Acceleration expressions per model, in the variables the generated script defines.
_RHS = {
    "pendulum": "dw = -(p['g'] / L) * np.sin(th)",
    "pendulum_small_angle": "dw = -(p['g'] / L) * th",
    "pendulum_linear_damping": "dw = -(p['g'] / L) * np.sin(th) - p['gamma'] * w",
    "pendulum_quadratic_damping": "dw = -(p['g'] / L) * np.sin(th) - p['beta'] * w * abs(w)",
    "harmonic": "dv = -p['omega'] ** 2 * (s - p['s0'])",
    "damped_harmonic": "dv = -p['omega'] ** 2 * (s - p['s0']) - 2 * p['zeta'] * p['omega'] * v",
    "free_fall": "ax, ay = 0.0, -p['g']",
    "linear_drag": "ax, ay = -p['k'] * vx, -p['g'] - p['k'] * vy",
    "quadratic_drag": "speed = np.hypot(vx, vy)\n    ax, ay = -p['c'] * speed * vx, -p['g'] - p['c'] * speed * vy",
}
_STATE = {
    "pendulum": ("th, w = state", "return [w, dw]", "theta_rad"),
    "oscillator": ("s, v = state", "return [v, dv]", "s_m"),
    "projectile": ("x, y, vx, vy = state", "return [vx, vy, ax, ay]", "x_m,y_m"),
}


def _param_lines(result: dict[str, Any]) -> list[str]:
    lines = []
    for p in result.get("parameters") or []:
        lines.append(f"    {p['name']} = {p['value']:.6g} +- {p['std']:.2g} {p['unit']}  (95%: {p['ci95'][0]:.5g} to {p['ci95'][1]:.5g})")
    return lines


def python_script(result: dict[str, Any], seconds: float | None = None) -> str:
    fs = result["fit_state"]
    spec = MODELS[fs["model_id"]]
    unpack, ret, columns = _STATE[spec.family]
    seconds = seconds or result["clip"]["frames"] / result["clip"]["fps"]
    length = f"L = {fs['ctx']['length_m']!r}  # m, radius of the measured swing arc\n" if spec.family == "pendulum" else ""
    status_line = "verified on held-out data" if result.get("status") == "accepted" else f"verdict: {result.get('status')} (see the app)"
    params = "\n".join(_param_lines(result)) or "    (no uncertainty intervals: the verdict was not a single accepted model)"
    return f'''"""{spec.title}, identified by PhysicsLens from a phone video ({status_line}).

Equation: {spec.latex}
Fitted parameters:
{params}

Simulates from the measured initial state and writes physicslens_trajectory.csv. Needs numpy and scipy.
"""

import numpy as np
from scipy.integrate import solve_ivp

p = {json.dumps(fs["params"], indent=4)}
{length}STATE0 = {json.dumps([float(v) for v in fs["y0"]])}  # measured initial state
FPS = {fs["fps"]!r}
SECONDS = {seconds!r}


def rhs(t, state):
    {unpack}
    {_RHS[spec.id]}
    {ret}


def simulate(seconds=SECONDS, fps=FPS, state0=STATE0):
    t = np.arange(int(seconds * fps)) / fps
    sol = solve_ivp(rhs, (t[0], t[-1]), state0, t_eval=t, method="DOP853", rtol=1e-7, atol=1e-9)
    return t, sol.y


if __name__ == "__main__":
    t, y = simulate()
    n = len(STATE0) // 2
    np.savetxt("physicslens_trajectory.csv", np.column_stack([t, y[:n].T]), delimiter=",",
               header="t_s,{columns}", comments="")
    print(f"Wrote {{len(t)}} samples to physicslens_trajectory.csv")
'''


def parameters_json(result: dict[str, Any]) -> str:
    fs = result["fit_state"]
    spec = MODELS[fs["model_id"]]
    doc = {
        "generator": "PhysicsLens",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": result.get("status"),
        "verdict_reason": result.get("reason"),
        "model": {"id": spec.id, "title": spec.title, "family": spec.family, "equation_latex": spec.latex},
        "parameters": result.get("parameters") or [{"name": k, "value": v} for k, v in fs["params"].items()],
        "geometry": fs["ctx"],
        "initial_state": fs["y0"],
        "calibration": result.get("calibration"),
        "units": "SI; angles in radians",
        "regimes": result.get("regimes"),
    }
    return json.dumps(doc, indent=2, default=str)
