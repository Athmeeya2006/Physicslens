"""Asteroid follow-up: an orbit from observations, how much one more night helps, and an ADES report.

- Observations are astrometric right ascension and declination (ICRF) from MPC observatory codes, each with its
  stated uncertainty. The observer's heliocentric position is the Earth's (astropy, TDB) plus the site's geocentric
  vector from the MPC's parallax constants.
- The orbit is a heliocentric state at an epoch, carried by Newton's pull of the Sun and the planets with the Moon
  (physics/nbody.py; positions from astropy's built-in ephemeris, ample for perturbations), with the light-time
  iterated for every observation. Fitted by weighted least squares with outliers beyond 3 sigma set aside (and
  reported), as orbit computers do.
- "Did my observation matter?": the 1-sigma position uncertainty a month ahead, from the fit's covariance, before
  and after adding a night. Checked by forecasting later observations and comparing the misses with the stated
  uncertainty.
- Reports in ADES PSV (version 2022, IAU ADES description of 16 March 2026), for the observer to submit with their
  own MPC observatory code. Never sent anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..physics.nbody import Body, propagate

C_AU_PER_DAY = 173.1446326846693
GM_SUN = 2.9591220828559115e-4  # AU^3/day^2, DE440
EARTH_RADIUS_KM = 6378.137
# km^3/s^2 (JPL DE440 values as used across PhysicsLens)
PLANET_GM = {"mercury": 22031.868551, "venus": 324858.592, "earth": 398600.435507, "moon": 4902.800118,
             "mars": 42828.375816, "jupiter": 126712764.1, "saturn": 37940584.8418, "uranus": 5794556.4,
             "neptune": 6836527.10058}
OUTLIER_SIGMA = 3.0


@dataclass
class Observations:
    t_utc_jd: np.ndarray
    ra: np.ndarray  # degrees
    dec: np.ndarray
    sigma_ra: np.ndarray  # arcsec (on the sky: already times cos dec)
    sigma_dec: np.ndarray
    stn: np.ndarray  # MPC codes
    obs_helio: np.ndarray  # observer's heliocentric position, AU (ICRF axes), (n, 3)

    def subset(self, m: np.ndarray) -> Observations:
        return Observations(*(getattr(self, k)[m] for k in ("t_utc_jd", "ra", "dec", "sigma_ra", "sigma_dec", "stn", "obs_helio")))


def parse_obscodes(text: str) -> dict[str, tuple[float, float, float]]:
    """The MPC's observatory list: code -> (east longitude deg, rho cos phi', rho sin phi') in Earth radii.
    Space-based and roving codes have no constants and are left out."""
    out = {}
    for ln in text.splitlines():
        if len(ln) < 30 or not ln[:3].strip() or ln.startswith("Code"):
            continue
        try:
            out[ln[:3]] = (float(ln[4:13]), float(ln[13:21]), float(ln[21:30]))
        except ValueError:
            continue
    return out


def observer_positions(stn: np.ndarray, t_utc_jd: np.ndarray, codes: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """Heliocentric positions (AU) of the observing sites at the observation times."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation, get_body_barycentric
    from astropy.time import Time

    t = Time(t_utc_jd, format="jd", scale="utc")
    earth = (get_body_barycentric("earth", t.tdb) - get_body_barycentric("sun", t.tdb)).xyz.to_value(u.au).T
    out = np.array(earth, float)
    for code in np.unique(stn):
        m = stn == code
        lon, rc, rs = codes[code]
        lam = math.radians(lon)
        loc = EarthLocation.from_geocentric(rc * math.cos(lam) * EARTH_RADIUS_KM * u.km, rc * math.sin(lam) * EARTH_RADIUS_KM * u.km,
                                            rs * EARTH_RADIUS_KM * u.km)
        pos, _ = loc.get_gcrs_posvel(t[m])
        out[m] += pos.xyz.to_value(u.au).T
    return out


def from_ades_rows(rows: list[dict[str, Any]], codes: dict[str, tuple[float, float, float]],
                   default_sigma: float = 1.0) -> Observations:
    """MPC observations as returned by its API (ADES fields), ground-based optical only."""
    from astropy.time import Time
    keep = [r for r in rows if r.get("stn") in codes and r.get("ra") and r.get("dec") and (r.get("Obstype") in (None, "optical"))]
    t = Time([r["obstime"].rstrip("Z") for r in keep], scale="utc").jd
    order = np.argsort(t)
    keep = [keep[i] for i in order]
    t = t[order]

    def num(v, d):
        try:
            return float(v) if v not in (None, "") else d
        except ValueError:
            return d

    stn = np.array([r["stn"] for r in keep])
    return Observations(t, np.array([float(r["ra"]) for r in keep]), np.array([float(r["dec"]) for r in keep]),
                        np.array([num(r.get("rmsra"), default_sigma) for r in keep]), np.array([num(r.get("rmsdec"), default_sigma) for r in keep]),
                        stn, observer_positions(stn, t, codes))


def planets(jd0: float, jd1: float, step: float = 0.25) -> list[Body]:
    """The planets and the Moon, heliocentric, tabulated every 6 hours over the span (and a margin)."""
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    grid = np.arange(jd0 - 2, jd1 + 2, step)
    t = Time(grid, format="jd", scale="tdb")
    sun = get_body_barycentric("sun", t)
    out = []
    for name, gm in PLANET_GM.items():
        p = (get_body_barycentric(name, t) - sun).xyz.to_value(u.au).T
        out.append(Body.from_table(name, gm, np.column_stack([grid, p])))
    return out


def tdb(t_utc_jd: np.ndarray) -> np.ndarray:
    from astropy.time import Time
    return Time(t_utc_jd, format="jd", scale="utc").tdb.jd


def predict(state0: np.ndarray, epoch_tdb: float, obs: Observations, bodies: list[Body]) -> tuple[np.ndarray, np.ndarray]:
    """Astrometric RA and Dec (degrees) for every observation: the light-time iterated twice."""
    t = tdb(obs.t_utc_jd)
    lo, hi = min(epoch_tdb, t.min() - 0.1), max(epoch_tdb, t.max())
    back = propagate(state0, epoch_tdb, lo, GM_SUN, bodies) if lo < epoch_tdb else None
    fwd = propagate(state0, epoch_tdb, hi, GM_SUN, bodies) if hi > epoch_tdb else None

    def at(tt):
        out = np.empty((len(tt), 6))
        m = tt < epoch_tdb
        if m.any():
            out[m] = back(tt[m]).T
        if (~m).any():
            out[~m] = fwd(tt[~m]).T
        return out

    tau = np.zeros(len(t))
    for _ in range(2):
        d = at(t - tau)[:, :3] - obs.obs_helio
        tau = np.linalg.norm(d, axis=1) / C_AU_PER_DAY
    d = at(t - tau)[:, :3] - obs.obs_helio
    n = np.linalg.norm(d, axis=1)
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360, np.degrees(np.arcsin(d[:, 2] / n))


def residuals(ra_c, dec_c, obs: Observations) -> tuple[np.ndarray, np.ndarray]:
    """Observed minus computed, arcsec on the sky (RA times cos dec)."""
    dra = ((obs.ra - ra_c + 180) % 360 - 180) * np.cos(np.radians(obs.dec)) * 3600
    return dra, (obs.dec - dec_c) * 3600


@dataclass
class OrbitSolution:
    state: np.ndarray
    epoch_tdb: float
    cov: np.ndarray
    used: np.ndarray  # mask of observations kept
    rms_arcsec: float
    chi2_per_dof: float


def fit(obs: Observations, state0: np.ndarray, epoch_tdb: float, bodies: list[Body], rounds: int = 3) -> OrbitSolution:
    """Weighted least squares on the angles; outliers beyond OUTLIER_SIGMA set aside, refitting up to `rounds` times."""
    from scipy.optimize import least_squares
    used = np.ones(len(obs.ra), bool)
    state = np.asarray(state0, float)
    steps = np.array([1e-8] * 3 + [1e-10] * 3)  # AU and AU/day: about 1.5 km and 2 mm/s

    for _ in range(rounds):
        sub = obs.subset(used)

        def resid(x, sub=sub):
            ra_c, dec_c = predict(x, epoch_tdb, sub, bodies)
            a, b = residuals(ra_c, dec_c, sub)
            return np.concatenate([a / sub.sigma_ra, b / sub.sigma_dec])

        def jac(x, resid=resid):
            base, cols = resid(x), []
            for i, h in enumerate(steps):
                q = x.copy()
                q[i] += h
                cols.append((resid(q) - base) / h)
            return np.column_stack(cols)

        sol = least_squares(resid, state, jac=jac, x_scale=steps * 1e4, max_nfev=30)
        state = sol.x
        ra_c, dec_c = predict(state, epoch_tdb, obs, bodies)
        a, b = residuals(ra_c, dec_c, obs)
        norm = np.hypot(a / obs.sigma_ra, b / obs.sigma_dec) / math.sqrt(2)
        new = norm <= OUTLIER_SIGMA
        if np.array_equal(new, used):
            break
        used = new
    else:
        # The last round still changed which observations count: refit once on that set, so the state and its
        # covariance describe the observations the solution keeps.
        sub = obs.subset(used)

        def resid(x, sub=sub):
            ra_c, dec_c = predict(x, epoch_tdb, sub, bodies)
            a, b = residuals(ra_c, dec_c, sub)
            return np.concatenate([a / sub.sigma_ra, b / sub.sigma_dec])

        def jac(x, resid=resid):
            base, cols = resid(x), []
            for i, h in enumerate(steps):
                q = x.copy()
                q[i] += h
                cols.append((resid(q) - base) / h)
            return np.column_stack(cols)

        sol = least_squares(resid, state, jac=jac, x_scale=steps * 1e4, max_nfev=30)
        state = sol.x
    J = sol.jac
    dof = max(len(sol.fun) - 6, 1)
    chi2 = float(sol.fun @ sol.fun) / dof
    cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2)
    ra_c, dec_c = predict(state, epoch_tdb, obs.subset(used), bodies)
    a, b = residuals(ra_c, dec_c, obs.subset(used))
    return OrbitSolution(state, epoch_tdb, cov, used, float(np.sqrt(np.mean(a ** 2 + b ** 2) / 2)), chi2)


def position_uncertainty(sol: OrbitSolution, t_utc_jd: float, observer_helio: np.ndarray, bodies: list[Body]) -> dict[str, float]:
    """The 1-sigma uncertainty ellipse of the predicted position at a time, as seen from a place (arcsec)."""
    one = Observations(np.array([t_utc_jd]), np.zeros(1), np.zeros(1), np.ones(1), np.ones(1), np.array(["500"]),
                       np.atleast_2d(observer_helio))
    base = np.array(predict(sol.state, sol.epoch_tdb, one, bodies)).ravel()
    cols = []
    for i, h in enumerate([1e-8] * 3 + [1e-10] * 3):
        q = sol.state.copy()
        q[i] += h
        p = np.array(predict(q, sol.epoch_tdb, one, bodies)).ravel()
        d = p - base
        cols.append([((d[0] + 180) % 360 - 180) * math.cos(math.radians(base[1])) * 3600 / h, d[1] * 3600 / h])
    J = np.array(cols).T
    C = J @ sol.cov @ J.T
    w = np.linalg.eigvalsh(C)
    return {"major_arcsec": float(math.sqrt(max(w[-1], 0))), "minor_arcsec": float(math.sqrt(max(w[0], 0))),
            "ra_dec": [float(base[0]), float(base[1])]}


def ades_psv(obs: Observations, rows: list[dict[str, Any]], mpc_code: str = "XXX", submitter: str = "Your Name",
             telescope: tuple[str, float, str] = ("reflector", 0.3, "CMOS")) -> str:
    """Observations as ADES PSV (version 2022): the context records, a keyword record and one data record each.
    The MPC code, names and telescope are placeholders for the observer's own; nothing is submitted."""
    head = ["# version=2022", "# observatory", f"! mpcCode {mpc_code}", "# submitter", f"! name {submitter}",
            "# observers", f"! name {submitter}", "# measurers", f"! name {submitter}", "# telescope",
            f"! design {telescope[0]}", f"! aperture {telescope[1]}", f"! detector {telescope[2]}"]
    keys = ["provID", "trkSub", "mode", "stn", "obsTime", "ra", "dec", "rmsRA", "rmsDec", "astCat", "mag", "band"]
    lines = [" | ".join(keys)]
    for r in rows:
        lines.append(" | ".join([str(r.get("provid") or ""), (str(r.get("trksub") or "pl1"))[:8], str(r.get("mode") or "CCD"), mpc_code,
                                 str(r["obstime"]), f"{float(r['ra']):.6f}", f"{float(r['dec']):+.6f}", str(r.get("rmsra") or "1.0"),
                                 str(r.get("rmsdec") or "1.0"), str(r.get("astcat") or "Gaia3"), str(r.get("mag") or ""), str(r.get("band") or "")]))
    return "\n".join(head + lines) + "\n"
