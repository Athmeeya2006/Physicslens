"""Sparse equation discovery with PySINDy: model-free evidence shown next to the hypothesis fits."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pysindy as ps

from ..sigproc.cleanup import differentiate
from .coords import Coordinates

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
    if family == "oscillator":
        return ps.GeneralizedLibrary(
            [_lib(_IDENTITY, (lambda x: x**3, lambda x: f"{x}^3"), bias=True), _lib(_IDENTITY, _ABS)],
            inputs_per_library=[[0], [1]],
        ), ["s", "s'"]
    if family == "projectile":
        drag = _lib(
            (lambda x, y: x * np.hypot(x, y), lambda x, y: f"|v|{x}"),
            (lambda x, y: y * np.hypot(x, y), lambda x, y: f"|v|{y}"),
        )
        return ps.GeneralizedLibrary(
            [_lib(_IDENTITY, bias=True), drag], inputs_per_library=[[2, 3], [2, 3]],
        ), ["x", "y", "x'", "y'"]
    raise ValueError(f"Unknown family {family!r}")


@dataclass(frozen=True)
class SindyResult:
    equations: dict[str, str]  # acceleration equations only, e.g. {"theta''": "-16.35 sin(theta) -0.08 theta'"}
    terms: dict[str, dict[str, float]]
    threshold: float


def discover(c: Coordinates, threshold: float = 0.1) -> SindyResult:
    fps = 1.0 / (c.t[1] - c.t[0])
    kin = differentiate(c.q, fps)
    d = c.q.shape[1]
    library, names = _library(c.family)
    model = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold, normalize_columns=True), feature_library=library)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # STLSQ warns when it prunes everything; that shows up as "0"
        model.fit(np.column_stack([kin.pos, kin.vel]), t=1.0 / fps,
                  x_dot=np.column_stack([kin.vel, kin.acc]), feature_names=names)

    features = model.get_feature_names()
    coefs = model.coefficients()
    equations, terms = {}, {}
    for i in range(d, 2 * d):  # acceleration rows; the first d rows only restate q' = q'
        active = {f: float(v) for f, v in zip(features, coefs[i], strict=True) if abs(v) > 0}
        lhs = f"{c.names[i - d]}''"
        terms[lhs] = active
        equations[lhs] = " ".join(f"{v:+.4g}" if f == "1" else f"{v:+.4g} {f}" for f, v in active.items()) or "0"
    return SindyResult(equations, terms, threshold)
