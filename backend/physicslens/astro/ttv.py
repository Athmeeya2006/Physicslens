"""A hidden planet from transit times: Le Verrier's method on an exoplanet.

1. Candidates. Near a first-order resonance j:j-1 the timing variations repeat with the super-period
   P^j = 1 / |j/P' - (j-1)/P| (Lithwick, Xie & Wu 2012, eq. psuper; P' the outer period, checked in the paper's
   source). A measured super-period therefore allows a few companion periods: for each j, outside or inside the
   transiting planet, on either side of the resonance. All are kept: the data, not a prior, choose.
2. N-body. Each candidate is fitted with a coplanar, edge-on three-body model integrated by REBOUND, transit times
   read where the transiting planet crosses the line of sight. Fitted: the transiting planet's reference time,
   period and eccentricity vector, the companion's period, mean longitude, eccentricity vector and mass.
3. Verdict. As every PhysicsLens law: fitted on the earlier 75% of transits, it must forecast the later 25%; masses
   are reported as a range over the fits that do, because mass and free eccentricity trade off (that degeneracy is
   shown, not hidden).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

EARTH_MASS_SUN = 3.0034896e-6
STEPS_PER_ORBIT = 60
ASSUMED_INNER_MASS_EARTH = 10.0  # the transiting planet's own mass barely changes its timing; stated, not fitted


def super_period(p_inner: float, p_outer: float, j: int) -> float:
    return 1.0 / abs(j / p_outer - (j - 1) / p_inner)


def companion_candidates(p_transiting: float, p_ttv: float, js: tuple[int, ...] = (2, 3, 4)) -> list[dict[str, Any]]:
    """Companion periods that give this super-period near each first-order resonance, outside or inside."""
    out = []
    for j in js:
        for sign in (1, -1):
            # Companion outside: P' = companion, P = transiting.  j/P' = (j-1)/P + sign/P^j
            inv = (j - 1) / p_transiting + sign / p_ttv
            if inv > 0:
                out.append({"j": j, "where": "outside", "period": j / inv, "resonance": f"{j}:{j - 1}"})
            # Companion inside: P' = transiting, P = companion.  (j-1)/P = j/P' - sign/P^j
            inv = j / p_transiting - sign / p_ttv
            if inv > 0:
                out.append({"j": j, "where": "inside", "period": (j - 1) / inv, "resonance": f"{j}:{j - 1}"})
    return out


@dataclass
class Setup:
    mstar: float  # solar masses
    epoch0: int  # the epoch the reference time belongs to


def _elements(k: float, h: float) -> tuple[float, float]:
    return min(math.hypot(k, h), 0.9), math.atan2(h, k)


def nbody_times(theta: np.ndarray, epochs: np.ndarray, s: Setup) -> np.ndarray:
    """Transit times (BJD) of the inner/transiting planet at the given epochs. theta =
    [T0, Pb, kb, hb, Pc, lam_c, kc, hc, mc_earth]; the observer looks along -y, so a transit is the planet crossing
    x = 0 on the observer's side (y > 0) of the star."""
    import rebound
    t0, pb, kb, hb, pc, lam_c, kc, hc, mc = theta
    sim = rebound.Simulation()
    sim.G = 2.959122082855911e-4  # AU^3 / (Msun day^2)
    sim.add(m=s.mstar)
    eb, pomb = _elements(kb, hb)
    ec, pomc = _elements(kc, hc)
    # The transiting planet starts exactly at transit: true longitude 90 degrees (on +y).
    sim.add(m=ASSUMED_INNER_MASS_EARTH * EARTH_MASS_SUN, P=pb, e=eb, pomega=pomb, f=math.pi / 2 - pomb)
    sim.add(m=max(mc, 0.0) * EARTH_MASS_SUN, P=pc, e=ec, pomega=pomc, l=lam_c)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = min(pb, pc) / STEPS_PER_ORBIT
    n_max = int(epochs.max() - s.epoch0)
    times = [0.0]
    star, planet = sim.particles[0], sim.particles[1]
    prev_x = 0.0
    t_end = (n_max + 0.6) * pb
    while sim.t < t_end and len(times) <= n_max:
        sim.steps(1)
        x, y = planet.x - star.x, planet.y - star.y
        if prev_x > 0 >= x and y > 0:
            vx = planet.vx - star.vx
            r3 = (x * x + y * y + (planet.z - star.z) ** 2) ** 1.5
            ax = -sim.G * (s.mstar) * x / r3
            # x(tau) = x + vx tau + ax tau^2 / 2 = 0 for tau in [-dt, 0]: second-order accurate at the crossing.
            if abs(ax) > 1e-30:
                disc = vx * vx - 2 * ax * x
                tau = (-vx + math.copysign(math.sqrt(max(disc, 0.0)), vx)) / ax if disc > 0 else -x / vx
                if not (-sim.dt * 1.01 <= tau <= 1e-9):
                    tau = -x / vx
            else:
                tau = -x / vx
            times.append(sim.t + tau)
        prev_x = x
    times_arr = np.array(times)
    idx = (epochs - s.epoch0).astype(int)
    out = np.full(len(epochs), np.nan)
    ok = idx < len(times_arr)
    out[ok] = t0 + times_arr[idx[ok]]
    return out


@dataclass
class CompanionFit:
    candidate: dict[str, Any]
    theta: np.ndarray
    cov: np.ndarray
    chi2: float
    bic: float
    heldout_chi2: float = math.nan
    heldout_rms_min: float = math.nan

    def summary(self) -> dict[str, Any]:
        names = ["T0", "Pb", "kb", "hb", "Pc", "lam_c", "kc", "hc", "mc_earth"]
        sig = np.sqrt(np.maximum(np.diag(self.cov), 0))
        return {"candidate": self.candidate, **{n: float(v) for n, v in zip(names, self.theta, strict=True)},
                **{f"{n}_sigma": float(v) for n, v in zip(names, sig, strict=True)},
                "eb": float(math.hypot(self.theta[2], self.theta[3])), "ec": float(math.hypot(self.theta[6], self.theta[7])),
                "chi2": self.chi2, "bic": self.bic, "heldout_chi2": self.heldout_chi2,
                "heldout_rms_min": self.heldout_rms_min}


def fit_companion(epochs: np.ndarray, times: np.ndarray, err: np.ndarray, candidate: dict[str, Any], s: Setup,
                  p_transiting: float, starts: tuple[float, ...] = (0.0, 1.57, 3.14, 4.71),
                  max_nfev: int = 300) -> CompanionFit | None:
    """The best N-body fit from a few starting phases for one candidate period (others stay within 3%)."""
    pc0 = candidate["period"]
    first = float(times[np.argmin(epochs)])
    best = None
    for lam in starts:
        for mc in (30.0, 300.0):
            p0 = np.array([first, p_transiting, 0.0, 0.0, pc0, lam, 0.0, 0.0, mc])
            lo = [first - 0.6, p_transiting * 0.995, -0.25, -0.25, pc0 * 0.97, -10, -0.25, -0.25, 0.1]
            hi = [first + 0.6, p_transiting * 1.005, 0.25, 0.25, pc0 * 1.03, 10, 0.25, 0.25, 5000]

            def resid(p: np.ndarray) -> np.ndarray:
                model = nbody_times(p, epochs, s)
                r = (model - times) / err
                return np.where(np.isfinite(r), r, 1e3)
            try:
                r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev, diff_step=1e-7)
            except (ValueError, RuntimeError):
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        return None
    chi2 = float(2 * best.cost)
    n, k = len(times), len(best.x)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * max(1.0, chi2 / max(n - k, 1))
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.inf)
    return CompanionFit(candidate, best.x, cov, chi2, chi2 + k * math.log(n))


def spread_starts(candidate: dict[str, Any], p_transiting: float, first: float, n: int,
                  seed: int = 0) -> list[np.ndarray]:
    """n starting points spread over what the timing data leave open: the companion's period across the candidate's
    window (the super-period fixes the distance to resonance only to a few percent), both eccentricity vectors up to
    0.2 in any direction, any phase, and masses from 10 to 1000 Earth masses on a log scale. A single fit from a few
    phases found one member of a family whose formal errors missed the published answer by many of themselves; the
    spread of the fits that forecast as well is the honest uncertainty."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        u = (i + rng.random()) / n  # stratified in period so the window is covered evenly
        eb, wb, ec, wc = 0.2 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi), 0.2 * math.sqrt(rng.random()), \
            rng.uniform(0, 2 * math.pi)
        out.append(np.array([first, p_transiting, eb * math.cos(wb), eb * math.sin(wb),
                             candidate["period"] * (0.97 + 0.06 * u), rng.uniform(0, 2 * math.pi),
                             ec * math.cos(wc), ec * math.sin(wc), 10 ** rng.uniform(1, 3)]))
    return out


def fit_from(p0: np.ndarray, epochs: np.ndarray, times: np.ndarray, err: np.ndarray, candidate: dict[str, Any],
             s: Setup, p_transiting: float, max_nfev: int = 300) -> CompanionFit | None:
    """One N-body fit from one starting point, in the same bounds as fit_companion."""
    pc0, first = candidate["period"], float(times[np.argmin(epochs)])
    lo = [first - 0.6, p_transiting * 0.995, -0.25, -0.25, pc0 * 0.97, -10, -0.25, -0.25, 0.1]
    hi = [first + 0.6, p_transiting * 1.005, 0.25, 0.25, pc0 * 1.03, 10, 0.25, 0.25, 5000]

    def resid(p: np.ndarray) -> np.ndarray:
        r = (nbody_times(p, epochs, s) - times) / err
        return np.where(np.isfinite(r), r, 1e3)
    try:
        best = least_squares(resid, np.clip(p0, lo, hi), bounds=(lo, hi), x_scale="jac", max_nfev=max_nfev,
                             diff_step=1e-7)
    except (ValueError, RuntimeError):
        return None
    chi2 = float(2 * best.cost)
    n, k = len(times), len(best.x)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac) * max(1.0, chi2 / max(n - k, 1))
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.inf)
    return CompanionFit(candidate, best.x, cov, chi2, chi2 + k * math.log(n))
