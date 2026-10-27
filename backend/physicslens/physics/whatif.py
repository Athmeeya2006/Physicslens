"""What-ifs: change a fitted parameter (or the pendulum length) and re-simulate from the same starting state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .coords import Coordinates
from .fit import simulate
from .models import MODELS

Operation = Literal["multiply", "set", "add"]
# Geometry a user can change besides the ODE parameters.
GEOMETRY_PARAMS = {"pendulum": {"L": "length_m"}}


class WhatIfError(ValueError):
    pass


@dataclass(frozen=True)
class Change:
    parameter: str
    operation: Operation
    value: float


def changeable(fit_state: dict[str, Any]) -> list[str]:
    spec = MODELS[fit_state["model_id"]]
    return [*spec.param_names, *GEOMETRY_PARAMS.get(spec.family, {})]


def apply(fit_state: dict[str, Any], changes: list[Change]) -> tuple[dict[str, float], dict[str, float]]:
    """New (params, ctx). Rejects unknown parameters and unphysical results instead of simulating nonsense."""
    spec = MODELS[fit_state["model_id"]]
    params = dict(fit_state["params"])
    ctx = dict(fit_state["ctx"])
    geometry = GEOMETRY_PARAMS.get(spec.family, {})
    bounds = {p.name: (p.lower, p.upper) for p in spec.params}
    for ch in changes:
        if ch.parameter in params:
            target, key = params, ch.parameter
        elif ch.parameter in geometry:
            target, key = ctx, geometry[ch.parameter]
        else:
            raise WhatIfError(f"Can't change {ch.parameter!r}; this model has {changeable(fit_state)}.")
        old = target[key]
        new = {"multiply": old * ch.value, "set": ch.value, "add": old + ch.value}[ch.operation]
        if not np.isfinite(new) or new < 0 or (key == "length_m" and new <= 0):
            raise WhatIfError(f"{ch.parameter} would become {new:.4g}, which isn't physical.")
        if key in bounds and not bounds[key][0] <= new <= bounds[key][1]:
            raise WhatIfError(f"{ch.parameter} = {new:.4g} is outside the range this model supports {bounds[key]}.")
        target[key] = float(new)
    return params, ctx


def simulate_whatif(fit_state: dict[str, Any], changes: list[Change], seconds: float) -> dict[str, Any]:
    """Ghost points in clip pixels for the changed system, starting from the measured initial state."""
    spec = MODELS[fit_state["model_id"]]
    params, ctx = apply(fit_state, changes)
    fps = fit_state["fps"]
    t = np.arange(int(seconds * fps)) / fps
    y0 = np.array(fit_state["y0"], dtype=float)
    q = simulate(spec, params, y0, t, ctx)
    if q is None:
        raise WhatIfError("The changed system could not be simulated (it diverged).")
    coords = Coordinates(spec.family, t, q, (), ctx)
    xy = coords.to_xy(q)
    ox, oy = fit_state["origin_px"]
    ppm = fit_state["px_per_m"]
    f0 = int(fit_state["first_frame"])
    points = [[f0 + i, round(ox + x * ppm, 2), round(oy - y * ppm, 2)] for i, (x, y) in enumerate(xy)]
    out: dict[str, Any] = {"params": params, "points": points}
    if spec.family == "pendulum":
        out["period_s"] = float(2 * np.pi * np.sqrt(ctx["length_m"] / params["g"]))  # small-angle estimate
        out["length_m"] = ctx["length_m"]
    elif spec.family == "oscillator":
        out["period_s"] = float(2 * np.pi / params["omega"])
    return out
