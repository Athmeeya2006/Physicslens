"""Sparse equation discovery with PySINDy: model-free evidence shown next to the hypothesis fits."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pysindy as ps

from ..sigproc.cleanup import differentiate
from .coords import Coordinates
from .fit import TRAIN_FRACTION, FitResult

# Plain lambdas, not numpy ufuncs: PySINDy inspects __code__ to count arguments.
_IDENTITY = (lambda x: x, lambda x: x)
_ABS = (lambda x: x * np.abs(x), lambda x: f"{x}|{x}|")


def _lib(*pairs, bias: bool = False) -> ps.CustomLibrary:
    return ps.CustomLibrary(library_functions=[f for f, _ in pairs], function_names=[n for _, n in pairs], include_bias=bias)


def _library(family: str) -> tuple[ps.GeneralizedLibrary, list[str]]:
    """Each term only sees the variables it can physically depend on.

    State columns are [q..., q'...]. Offering both theta and sin(theta) (or positions to a projectile)
    makes near-collinear columns that sparse regression splits into nonsense.
    """
    if family == "pendulum":
        return ps.GeneralizedLibrary(
            [_lib((lambda x: np.sin(x), lambda x: f"sin({x})")), _lib(_IDENTITY, _ABS)],
            inputs_per_library=[[0], [1]],
        ), ["theta", "theta'"]
    if family in ("oscillator", "line"):
        return ps.GeneralizedLibrary(
            [_lib(_IDENTITY, (lambda x: x**3, lambda x: f"{x}^3"), bias=True), _lib(_IDENTITY, _ABS)],
            inputs_per_library=[[0], [1]],
        ), ["s", "s'"]
    if family in ("projectile", "plane"):
        drag = _lib(
            (lambda x, y: x * np.hypot(x, y), lambda x, y: f"|v|{x}"),
            (lambda x, y: y * np.hypot(x, y), lambda x, y: f"|v|{y}"),
        )
        return ps.GeneralizedLibrary(
            [_lib(_IDENTITY, bias=True), drag], inputs_per_library=[[2, 3], [2, 3]],
        ), ["x", "y", "x'", "y'"]
    raise ValueError(f"Unknown family {family!r}")


# The term that must dominate each family's acceleration for the regression to be describing the same physics as
# the verified law: gravity's restoring torque, the spring force, and gravity itself for a flight.
LEADING_TERM = {"pendulum": ("sin(theta)",), "oscillator": ("s",), "projectile": ("1",)}
# Shown only if it forecasts the held-out motion nearly as well as the verified law: within 2 times its error, but
# never stricter than 5% of the motion, since 2 times a near-noise-level error would hide a regression that is fine.
MAX_HELDOUT_RATIO = 2.0
HELDOUT_FLOOR = 0.05


@dataclass(frozen=True)
class SindyResult:
    equations: dict[str, str]  # acceleration equations only, e.g. {"theta''": "-16.35 sin(theta) -0.08 theta'"}
    terms: dict[str, dict[str, float]]
    threshold: float
    contributions: dict[str, dict[str, float]] = field(default_factory=dict)  # RMS size of each term


def _fit(c: Coordinates, threshold: float, n: int | None = None) -> tuple[ps.SINDy, np.ndarray]:
    fps = 1.0 / (c.t[1] - c.t[0])
    kin = differentiate(c.q[:n], fps)
    library, names = _library(c.family)
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold, normalize_columns=True), feature_library=library)
    x = np.column_stack([kin.pos, kin.vel])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # STLSQ warns when it prunes everything; that shows up as "0"
        model.fit(x, t=1.0 / fps, x_dot=np.column_stack([kin.vel, kin.acc]), feature_names=names)
    return model, x


def discover(c: Coordinates, threshold: float = 0.1) -> SindyResult:
    d = c.q.shape[1]
    model, x = _fit(c, threshold)

    features = model.get_feature_names()
    coefs = model.coefficients()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta = np.asarray(model.feature_library.transform(x))
    rms = np.sqrt(np.mean(theta**2, axis=0))
    equations, terms, contributions = {}, {}, {}
    for i in range(d, 2 * d):  # acceleration rows; the first d rows only restate q' = q'
        active = {f: float(v) for f, v in zip(features, coefs[i], strict=True) if abs(v) > 0}
        lhs = f"{c.names[i - d]}''"
        terms[lhs] = active
        contributions[lhs] = {f: abs(v) * float(rms[features.index(f)]) for f, v in active.items()}
        equations[lhs] = " ".join(f"{v:+.4g}" if f == "1" else f"{v:+.4g} {f}" for f, v in active.items()) or "0"
    return SindyResult(equations, terms, threshold, contributions)


def heldout_nrmse(c: Coordinates, threshold: float = 0.1) -> float | None:
    """Fit on the same first part as the verified models and forecast the rest, scored the same way (RMS / std(q))."""
    n_train = int(len(c.t) * TRAIN_FRACTION)
    if n_train < 10 or n_train >= len(c.t) - 2:
        return None
    model, _ = _fit(c, threshold, n_train)
    # Start from the measured state where the held-out part begins: a raw differentiated state at t = 0 would
    # charge the regression for ten seconds of phase drift from one noisy velocity sample.
    fps = 1.0 / (c.t[1] - c.t[0])
    kin = differentiate(c.q, fps)
    x0 = np.concatenate([kin.pos[n_train], kin.vel[n_train]])
    t = c.t[n_train:] - c.t[n_train]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            sim = model.simulate(x0, t)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
    d = c.q.shape[1]
    if sim.shape[0] != len(t) or not np.all(np.isfinite(sim)):
        return None
    err = sim[:, :d] - c.q[n_train:]
    return float(np.sqrt(np.mean(err**2)) / (np.std(c.q) + 1e-12))


def vet(c: Coordinates, accepted: FitResult | None, threshold: float = 0.1) -> dict:
    """Decide whether the model-free regression may be shown next to the verified law.

    It is shown only when it forecasts the held-out motion within MAX_HELDOUT_RATIO of the verified law's error and
    its largest term is the family's leading physics; otherwise it is kept in the JSON but hidden, because a sparse
    fit to differentiated noise reads like a competing equation when it is not one.
    """
    found = discover(c, threshold)
    out = {"equations": found.equations, "shown": False, "heldout_nrmse": None, "reason": ""}
    if accepted is None:
        out["reason"] = "no verified law to compare with"
        return out
    nrmse = heldout_nrmse(c, threshold)
    out["heldout_nrmse"] = nrmse
    leading = LEADING_TERM.get(c.family)
    # The last acceleration row carries the family's physics (y'' for a flight, the only row otherwise).
    last = list(found.contributions.values())[-1] if found.contributions else {}
    agrees = leading is not None and bool(last) and max(last, key=last.get) in leading
    if nrmse is None:
        out["reason"] = "its forecast of the held-out motion diverged"
    elif nrmse > max(MAX_HELDOUT_RATIO * accepted.heldout_nrmse, HELDOUT_FLOOR):
        out["reason"] = (f"it forecasts the held-out motion with {nrmse:.1%} error, against {accepted.heldout_nrmse:.1%} "
                         "for the verified law")
    elif not agrees:
        out["reason"] = "its largest term is not the physics the verified law found"
    else:
        out["shown"] = True
    return out
