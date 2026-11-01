"""What-ifs: change a fitted parameter (or the pendulum length) and re-simulate from the same starting state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .coords import Coordinates
from .custom import spec_for
from .fit import simulate

Operation = Literal["multiply", "set", "add"]
# Geometry a user can change besides the ODE parameters.
GEOMETRY_PARAMS = {"pendulum": {"L": "length_m"}}

# The object itself: its mass (same size, a denser material) and its size (same material and shape, so mass goes as
# size cubed). The fitted parameters are per unit mass or per moment of inertia, so each scales as m^a s^b with the
# exponents below; anything not listed (g, sliding friction f = mu g, rest positions) does not depend on the object.
#   spring        omega = sqrt(k/m), zeta = b / (2 sqrt(k m)), duffing beta = k3/m
#   air drag      linear (Stokes) k = 6 pi eta R / m ~ s/m; quadratic c = rho Cd A / (2m) ~ s^2/m
#   pendulum      pivot and air damping per unit mass: gamma ~ 1/m; quadratic air beta ~ A/m ~ s^2/m
#   spinning      I ~ m s^2: dry c = (mu m g r)/I ~ 1/s; bearing k = b/I; air q ~ rho s^5 / I; torsion kappa = K/I
OBJECT_PARAMS = {"object_mass": "mass of the moving object (same size)",
                 "object_size": "size of the moving object (same material, so its mass grows as size cubed)"}
OBJECT_EXPONENTS: dict[str, dict[str, tuple[float, float]]] = {
    "oscillator": {"omega": (-0.5, 0.0), "zeta": (-0.5, 0.0), "beta": (-1.0, 0.0)},
    "projectile": {"k": (-1.0, 1.0), "c": (-1.0, 2.0)},
    "pendulum": {"gamma": (-1.0, 0.0), "beta": (-1.0, 2.0)},
    "rotation": {"c": (0.0, -1.0), "k": (-1.0, -2.0), "q": (-1.0, 3.0), "kappa": (-1.0, -2.0)},
    "double_pendulum": {"gamma": (-1.0, 0.0)},
}


class WhatIfError(ValueError):
    pass


@dataclass(frozen=True)
class Change:
    parameter: str
    operation: Operation
    value: float


def changeable(fit_state: dict[str, Any]) -> list[str]:
    spec = spec_for(fit_state)
    return [*spec.param_names, *GEOMETRY_PARAMS.get(spec.family, {}), *OBJECT_PARAMS]


def object_factors(family: str, param_names: list[str], mass: float, size: float) -> dict[str, float]:
    """Each fitted parameter's factor when the object's mass is multiplied by `mass` and its size by `size` (at the
    same material, so mass also grows by size cubed)."""
    m = mass * size ** 3
    ex = OBJECT_EXPONENTS.get(family, {})
    return {n: float(m ** ex[n][0] * size ** ex[n][1]) for n in param_names if n in ex}


def object_note(family: str, factors: dict[str, float], mass: float, size: float) -> str:
    """Plain words on what the change does to this law, including when it does nothing (that is physics too)."""
    what = " and ".join(x for x in ((f"{mass:g} times the mass" if mass != 1 else ""),
                                    (f"{size:g} times the size" if size != 1 else "")) if x) or "the same object"
    moved = {n: f for n, f in factors.items() if abs(f - 1) > 1e-9}
    if not moved:
        why = {"pendulum": "a pendulum's swing does not depend on the bob's mass (gravity and inertia both scale with it)",
               "projectile": "without air drag every object falls the same way (Galileo)",
               "oscillator": "nothing in this law depends on it",
               "line": "this law has no force term that depends on the object",
               "plane": "this law has no force term that depends on the object"}.get(family, "nothing in this law depends on it")
        return f"With {what}, the motion is the same: {why}."
    parts = ", ".join(f"{n} x{f:.3g}" for n, f in moved.items())
    return f"With {what}: {parts} (each parameter is per unit mass or inertia, so it scales with the object)."


def apply(fit_state: dict[str, Any], changes: list[Change]) -> tuple[dict[str, float], dict[str, float]]:
    """New (params, ctx). Rejects unknown parameters and unphysical results instead of simulating nonsense."""
    spec = spec_for(fit_state)
    params = dict(fit_state["params"])
    ctx = dict(fit_state["ctx"])
    geometry = GEOMETRY_PARAMS.get(spec.family, {})
    bounds = {p.name: (p.lower, p.upper) for p in spec.params}
    scale = {"object_mass": 1.0, "object_size": 1.0}
    for ch in changes:
        if ch.parameter in OBJECT_PARAMS:
            if ch.operation != "multiply" or not np.isfinite(ch.value) or not 0.01 <= ch.value <= 100:
                raise WhatIfError(f"{ch.parameter} can only be multiplied, by 0.01 to 100.")
            scale[ch.parameter] *= ch.value
            continue
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
    for name, f in object_factors(spec.family, list(params), scale["object_mass"], scale["object_size"]).items():
        new = params[name] * f
        if name in bounds and not bounds[name][0] <= new <= bounds[name][1]:
            raise WhatIfError(f"{name} = {new:.4g} is outside the range this model supports {bounds[name]}.")
        params[name] = float(new)
    return params, ctx


def simulate_whatif(fit_state: dict[str, Any], changes: list[Change], seconds: float) -> dict[str, Any]:
    """Ghost points in clip pixels for the changed system, starting from the measured initial state."""
    spec = spec_for(fit_state)
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
    mass = float(np.prod([c.value for c in changes if c.parameter == "object_mass"] or [1.0]))
    size = float(np.prod([c.value for c in changes if c.parameter == "object_size"] or [1.0]))
    if mass != 1 or size != 1:
        out["note"] = object_note(spec.family, object_factors(spec.family, spec.param_names, mass, size), mass, size)
    if spec.family == "pendulum":
        out["period_s"] = float(2 * np.pi * np.sqrt(ctx["length_m"] / params["g"]))  # small-angle estimate
        out["length_m"] = ctx["length_m"]
    elif spec.family == "oscillator":
        out["period_s"] = float(2 * np.pi / params["omega"])
    return out
