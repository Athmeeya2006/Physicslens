"""Predictions from a law found in a video, with and without the planets added as known forcing.

What was discovered (the fitted GM of the Sun and the fitted starting state, in the video's plane) is kept apart from
what is added (the planets: positions from our engine seeded with JPL states, masses from JPL). A result made this
way says "planets added from JPL, not discovered from the video".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ephem import J2000, catalogue_system
from .nbody import Body, Options, Prescribed, simulate

PLANETS = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres"]
PROVENANCE = "planets added from JPL, not discovered from the video"


def lift(fit_state: dict[str, Any], render_meta: dict[str, Any]) -> tuple[float, np.ndarray]:
    """(Julian date, 3D heliocentric state) of the fitted starting point: the video's 2D plane back into 3D."""
    basis = np.asarray(render_meta["basis"], dtype=float)
    y0 = np.asarray(fit_state["y0"], dtype=float)
    pos = basis[:2].T @ y0[:2]
    vel = basis[:2].T @ y0[2:4]
    jd0 = float(render_meta["start_jd"]) + int(fit_state["first_frame"]) * float(render_meta["step_days"])
    return jd0, np.concatenate([pos, vel])


def planets_at(jd0: float) -> list[Body]:
    """Every planet (and the Moon, Pluto, Ceres) at jd0: JPL's J2000 states carried there by our engine."""
    bodies = catalogue_system(PLANETS)
    if abs(jd0 - J2000) < 1e-9:
        return bodies
    run = simulate(catalogue_gm(), bodies, J2000, np.array([jd0]), Options(max_seconds=900))
    return [Body(b.name, b.gm, np.concatenate([run.pos[0, i], run.vel[0, i]])) for i, b in enumerate(bodies)]


def catalogue_gm() -> float:
    from .ephem import sun_gm
    return sun_gm()


def planet_paths(jd0: float, jd1: float) -> tuple[Prescribed, ...]:
    """The planets as known forcing between jd0 and jd1: their real paths (JPL's masses, JPL's Sun), as splines
    through daily positions, so the target's own discovered Sun never moves them."""
    from scipy.interpolate import CubicSpline
    lo, hi = min(jd0, jd1) - 2, max(jd0, jd1) + 2
    days = np.arange(lo, hi, 1.0)
    out = []
    start = planets_at(jd0)
    fwd = days[days >= jd0]
    back = days[days < jd0][::-1]
    pieces = []
    for grid in (back, fwd):
        if len(grid):
            pieces.append(simulate(catalogue_gm(), [Body(b.name, b.gm, b.state.copy()) for b in start], jd0, grid,
                                   Options(max_seconds=900)))
    jd_all = np.concatenate([r.jd for r in pieces])
    pos_all = np.concatenate([r.pos for r in pieces])
    order = np.argsort(jd_all)
    jd_all, pos_all = jd_all[order], pos_all[order]
    for i, b in enumerate(start):
        out.append(Prescribed(b.name, b.gm, CubicSpline(jd_all, pos_all[:, i])))
    return tuple(out)


def propagate(gm_sun: float, state: np.ndarray, jd0: float, jd_eval: np.ndarray, planets: bool,
              gr: bool = False, extra=None, prescribed: tuple[Prescribed, ...] | None = None) -> np.ndarray:
    """The target's heliocentric positions at jd_eval: the discovered Sun (gm_sun) alone, or with the planets as
    known forcing (prescribed paths: they pull the target, and the Sun through the indirect term)."""
    jd_eval = np.asarray(jd_eval, float)
    if prescribed is None:
        prescribed = planet_paths(jd0, float(jd_eval[-1])) if planets else ()
    run = simulate(gm_sun, [Body("target", 0.0, np.asarray(state, float), massive=False)], jd0, jd_eval,
                   Options(gr=gr, prescribed=prescribed if planets else (), extra=extra, max_seconds=900))
    return run.of("target")[0]


def law_extra(fit_state: dict[str, Any], render_meta: dict[str, Any]):
    """A fitted law's terms beyond the Sun's pull, as an extra acceleration in 3D (None when there are none). Both
    small terms the Apophis laws have carried act along the velocity in the video's plane, so here along the 3D
    velocity as well: a velocity term b (-b v) or a constant push A0 along the motion (A0 v / |v|)."""
    params = fit_state["params"]
    b, a0 = params.get("b"), params.get("A0")
    if b:
        return lambda r, v: -float(b) * v
    if a0:
        return lambda r, v: float(a0) * v / np.linalg.norm(v, axis=-1, keepdims=True)
    return None


def track_coordinates(result: dict[str, Any]):
    """The stored track in the fit's own units and origin (the Sun's pixel), as the fit saw it."""
    from ..physics.coords import Coordinates
    from ..sigproc.trajectory import from_pixels
    fs = result["fit_state"]
    pts = np.array(result["real_track_px"], dtype=float)
    traj = from_pixels(pts[:, 0].astype(int), pts[:, 1:3], float(fs["fps"]), float(fs["px_per_m"]),
                       origin=np.asarray(fs["origin_px"], float))
    return Coordinates("plane", traj.t, traj.xy, ("x", "y"), dict(fs.get("ctx") or {}))


def state_covariance(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(mean, covariance) of [GM, x0, y0, vx0, vy0] from the fit to the stored track: sigma^2 (J^T J)^-1 with the
    Jacobian by finite differences and sigma from the residuals. Linear, so it suits the small spread of a good fit."""
    from ..physics.custom import spec_for
    from ..physics.fit import simulate as fit_sim
    fs = result["fit_state"]
    c = track_coordinates(result)
    spec = spec_for(fs)
    params = dict(fs["params"])
    theta = np.array([params["mu"], *fs["y0"]], dtype=float)

    def model(th: np.ndarray) -> np.ndarray:
        sim = fit_sim(spec, {**params, "mu": th[0]}, th[1:], c.t, c.ctx)
        return sim.ravel()

    base = model(theta)
    sigma2 = float(np.mean((base - c.q.ravel()) ** 2))
    jac = np.empty((base.size, len(theta)))
    for k in range(len(theta)):
        h = 1e-6 * max(abs(theta[k]), 1e-9)
        step = np.zeros_like(theta)
        step[k] = h
        jac[:, k] = (model(theta + step) - model(theta - step)) / (2 * h)
    return theta, sigma2 * np.linalg.inv(jac.T @ jac)
