"""Stellar occultations: when the star vanished and came back, and the asteroid's outline from many observers.

- Edge times from a light curve: the star's light drops to the asteroid's own level at disappearance and returns
  at reappearance; each frame integrates over its exposure, so an edge inside a frame gives a partial drop. Fitted
  by least squares, with one-sigma errors from where chi-square rises by one (a profile in each edge time).
  Diffraction is not modelled (its scale, the Fresnel length sqrt(lambda D / 2), is about 1 km for an outer-planet
  moon and far less for a main-belt asteroid: below most cameras' time resolution).
- Chords: each observer's position projected onto the plane perpendicular to the star (the fundamental plane,
  axes f to the east and g to the north), minus the body's own position projected the same way, at the edge times.
  Every positive chord gives two points on the limb, each with its time error turned into km by the shadow's speed.
- The outline: a circle (centre and radius) or an ellipse (centre, axes, angle) fitted to the limb points, weighted
  by their errors. A negative chord (an observer who saw no occultation) must stay outside: a fit whose outline
  crosses a miss is penalised, so misses constrain the shape too.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

# --- the light curve -------------------------------------------------------------------------------------------------


def _model(t: np.ndarray, exp: float, t_d: float, t_r: float, high: float, low: float) -> np.ndarray:
    """Each frame's mean brightness over [t - exp/2, t + exp/2] with the star hidden between t_d and t_r."""
    a, b = t - exp / 2, t + exp / 2
    hidden = np.clip(np.minimum(b, t_r) - np.maximum(a, t_d), 0, None) / exp
    return high - (high - low) * hidden


def _levels(h: np.ndarray, flux: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Given the hidden fraction of every frame, the two levels are linear: flux = high (1 - h) + low h."""
    A = np.column_stack([1 - h, h]) * w[:, None]
    (high, low), *_ = np.linalg.lstsq(A, flux * w, rcond=None)
    r = (A @ [high, low] - flux * w)
    return float(high), float(low), float(r @ r)


def edge_times(t: np.ndarray, flux: np.ndarray, exposure: float, sigma: float | None = None,
               resolution: int = 40) -> dict[str, Any]:
    """Disappearance and reappearance times (and the levels) from a light curve, with one-sigma errors.

    The integrated model is flat while an edge moves between two frames' exposures, so instead of following a
    gradient, both edge times are scanned on a fine grid (resolution steps per exposure, over two frames either
    side of the first and last dark frame); the levels are solved exactly at each point. Each edge's error is where
    chi-square, minimised over the other edge, rises by one."""
    t, flux = np.asarray(t, float), np.asarray(flux, float)
    # Two levels from a two-cluster split (not percentiles or the median: the star may be hidden for a few frames of a
    # long recording, or for most of it), and the event is the longest run of dark frames (gaps of up to two frames
    # bridged), so a noise frame elsewhere cannot stretch it.
    lo_lvl, hi_lvl = float(np.min(flux)), float(np.max(flux))
    for _ in range(50):
        mid = (lo_lvl + hi_lvl) / 2
        dark, bright = flux[flux < mid], flux[flux >= mid]
        if not len(dark) or not len(bright):
            break
        lo_lvl, hi_lvl = float(np.mean(dark)), float(np.mean(bright))
    noise = sigma or float(1.4826 * np.median(np.abs(np.diff(flux) - np.median(np.diff(flux)))) / np.sqrt(2)) or 1e-9
    if hi_lvl - lo_lvl < 5 * noise:
        raise ValueError("No drop in the light curve: no occultation seen (a negative chord).")
    dark_idx = np.flatnonzero(flux < (lo_lvl + hi_lvl) / 2)
    runs, start = [], dark_idx[0]
    for a, b in itertools.pairwise(dark_idx):
        if b - a > 3:
            runs.append((start, a))
            start = b
    runs.append((start, dark_idx[-1]))
    i0, i1 = max(runs, key=lambda r: r[1] - r[0])
    low_frames = np.zeros(len(flux), bool)
    low_frames[i0:i1 + 1] = True
    sig = sigma or float(np.std(flux[~low_frames])) or 1.0
    w = np.full(len(t), 1 / sig)
    gd = np.linspace(t[i0] - 2 * exposure, t[i0] + exposure, 3 * resolution + 1)
    gr = np.linspace(t[i1] - exposure, t[i1] + 2 * exposure, 3 * resolution + 1)
    a, b = t - exposure / 2, t + exposure / 2
    chi = np.full((len(gd), len(gr)), np.inf)
    for i, td in enumerate(gd):
        for j, tr in enumerate(gr):
            if tr <= td:
                continue
            h = np.clip(np.minimum(b, tr) - np.maximum(a, td), 0, None) / exposure
            chi[i, j] = _levels(h, flux, w)[2]
    i, j = np.unravel_index(np.argmin(chi), chi.shape)
    best = float(chi[i, j])

    def width(profile: np.ndarray, grid: np.ndarray, k: int) -> float:
        ok = np.flatnonzero(profile - best <= 1.0)
        lo, hi = grid[ok.min()], grid[ok.max()]
        step = grid[1] - grid[0]
        return float(max(hi - lo + step, step) / 2)  # half the width of the region within one unit of the best

    h = np.clip(np.minimum(b, gr[j]) - np.maximum(a, gd[i]), 0, None) / exposure
    high, low, _ = _levels(h, flux, w)
    return {"disappear": float(gd[i]), "reappear": float(gr[j]),
            "disappear_sigma": width(chi.min(axis=1), gd, i), "reappear_sigma": width(chi.min(axis=0), gr, j),
            "duration": float(gr[j] - gd[i]), "drop": float(1 - low / high) if high else float("nan"), "chi2": best, "n": len(t)}


# --- the fundamental plane --------------------------------------------------------------------------------------------

def plane_axes(star_ra_deg: float, star_dec_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors f (east) and g (north) of the plane perpendicular to the star's direction."""
    a, d = math.radians(star_ra_deg), math.radians(star_dec_deg)
    f = np.array([-math.sin(a), math.cos(a), 0.0])
    g = np.array([-math.sin(d) * math.cos(a), -math.sin(d) * math.sin(a), math.cos(d)])
    return f, g


def project(vec_km: np.ndarray, axes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    f, g = axes
    return np.column_stack([vec_km @ f, vec_km @ g])


@dataclass
class Chord:
    observer: str
    positive: bool
    points: np.ndarray  # positive: [[f, g] at disappearance, [f, g] at reappearance]; negative: the track's points
    sigma_km: np.ndarray  # per point, along the chord


# --- the outline -------------------------------------------------------------------------------------------------------

def _ellipse_r(theta: np.ndarray, a: float, b: float, phi: float) -> np.ndarray:
    """Radius of an ellipse (semi-axes a, b, a's direction phi from f toward g) in direction theta."""
    c = np.cos(theta - phi)
    s = np.sin(theta - phi)
    return a * b / np.sqrt((b * c) ** 2 + (a * s) ** 2)


def _miss_penalty(chords: list[Chord], fc: float, gc: float, a: float, b: float, phi: float) -> np.ndarray:
    """How far inside the outline any negative chord's track goes (km; zero when it stays outside)."""
    out = []
    for ch in chords:
        if ch.positive:
            continue
        d = ch.points - [fc, gc]
        r = np.hypot(d[:, 0], d[:, 1])
        inside = _ellipse_r(np.arctan2(d[:, 1], d[:, 0]), a, b, phi) - r
        out.append(max(0.0, float(inside.max())))
    return np.array(out)


def fit_outline(chords: list[Chord], shape: str = "circle", miss_sigma_km: float = 1.0) -> dict[str, Any]:
    """Weighted fit of a circle or an ellipse to the limb points of the positive chords, misses as constraints.
    Returns the parameters with one-sigma errors, the residual of every limb point (observed minus calculated, km,
    positive outside) and the chi-square."""
    pts = np.vstack([c.points for c in chords if c.positive])
    sig = np.concatenate([c.sigma_km for c in chords if c.positive])
    fc0, gc0 = pts.mean(axis=0)
    r0 = float(np.mean(np.hypot(*(pts - [fc0, gc0]).T)))

    def unpack(p):
        if shape == "circle":
            return p[0], p[1], p[2], p[2], 0.0
        return p[0], p[1], p[2], p[3], p[4]

    def resid(p):
        fc, gc, a, b, phi = unpack(p)
        d = pts - [fc, gc]
        r = np.hypot(d[:, 0], d[:, 1])
        oc = r - _ellipse_r(np.arctan2(d[:, 1], d[:, 0]), a, b, phi)
        return np.concatenate([oc / sig, _miss_penalty(chords, fc, gc, a, b, phi) / miss_sigma_km])

    x0 = [fc0, gc0, r0] if shape == "circle" else [fc0, gc0, r0 * 1.02, r0 * 0.98, 0.3]
    sol = least_squares(resid, x0)
    n_pts = len(pts)
    k = len(x0)
    chi2 = float(sol.fun[:n_pts] @ sol.fun[:n_pts])
    J = sol.jac[:n_pts]
    cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2 / max(n_pts - k, 1))
    fc, gc, a, b, phi = unpack(sol.x)
    names = ["fc", "gc", "radius"] if shape == "circle" else ["fc", "gc", "a", "b", "phi"]
    out = {"shape": shape, "chi2": chi2, "dof": n_pts - k, "points": n_pts,
           "params": {n: float(v) for n, v in zip(names, sol.x, strict=True)},
           "sigma": {n: float(math.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)},
           "oc_km": [float(x) for x in sol.fun[:n_pts] * sig],
           "misses_crossed": int(np.sum(_miss_penalty(chords, fc, gc, a, b, phi) > 3 * miss_sigma_km))}
    if shape == "ellipse":
        if out["params"]["b"] > out["params"]["a"]:  # report a as the semi-major axis
            p, s = out["params"], out["sigma"]
            p["a"], p["b"], s["a"], s["b"] = p["b"], p["a"], s["b"], s["a"]
            p["phi"] += math.pi / 2
        out["params"]["phi"] = float(out["params"]["phi"] % math.pi)
        out["equivalent_radius"] = float(math.sqrt(out["params"]["a"] * out["params"]["b"]))
    return out
