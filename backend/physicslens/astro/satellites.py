"""Satellites: which object it is, whether it is where its catalogue orbit says, and whether it moved.

- Catalogue orbits are CelesTrak's GP elements (OMM), propagated with SGP4 (the model they are fitted for) and
  turned from TEME into GCRS by astropy. Reference trajectories (NASA's ISS ephemeris, an OEM file) are read as
  given (EME2000, the same axes as GCRS to within a few milliarcseconds).
- Offsets between two trajectories are split radial / in-track / cross-track; an in-track offset is also a time
  offset (km divided by speed): about 7.7 km per second in low orbit, which is why the clock matters.
- Observations are topocentric right ascension and declination from a site, as a camera with plate solving gives.
- A manoeuvre is a law, not a hunch: one orbit (two-body + J2 + an along-track drag term, integrated) against the
  same orbit with an impulse at an unknown time between passes. The impulse is accepted only by the usual rule:
  it must forecast the held-out passes and win by BIC.
- Tumbling: a brightness series' periodogram, with the half-period alias checked (a symmetric body flashes twice
  per turn).
- Reports in the IOD format (G. Lewis, 1998, clarified 2002; satobs.org/position/IODformat.html), for the observer
  to submit themselves.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km, equatorial (WGS84)
# Zonal harmonics of the Earth's gravity (EGM2008, unnormalised): the flattening and the next two terms. Over a few
# days J4 alone moves a low orbit's plane by kilometres, far more than a camera's 30 arcseconds.
J2 = 1.08262668e-3
J3 = -2.53265648533e-6
J4 = -1.61962159137e-6
TRAIN_FRACTION = 0.75
DECISIVE_DELTA_BIC = 10.0
HELDOUT_CHI2 = 4.0
RELATIVE_FORECAST = 4.0  # chi-square ratio (a halved miss) an impulse needs when one orbit cannot forecast at all


# --- trajectories ------------------------------------------------------------------------------------------------

@dataclass
class Trajectory:
    t_jd: np.ndarray  # UTC Julian dates
    r: np.ndarray  # km, GCRS axes, (n, 3)
    v: np.ndarray  # km/s


def read_oem(text: str) -> Trajectory:
    """A CCSDS OEM ephemeris (NASA's ISS file): epoch, position (km), velocity (km/s) per line."""
    from astropy.time import Time
    rows = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) == 7 and parts[0][:2].isdigit() and "T" in parts[0]:
            rows.append(parts)
    if not rows:
        raise ValueError("No state lines found: not an OEM ephemeris.")
    t = Time([r[0] for r in rows], scale="utc").jd
    a = np.array([[float(x) for x in r[1:]] for r in rows])
    return Trajectory(np.asarray(t), a[:, :3], a[:, 3:])


def satrec_from_omm(omm: dict[str, Any]):
    from sgp4 import omm as sgp4_omm
    from sgp4.api import Satrec
    sat = Satrec()
    sgp4_omm.initialize(sat, {k: str(v) for k, v in omm.items()})
    return sat


def propagate_gp(omm: dict[str, Any], t_jd: np.ndarray) -> Trajectory:
    """SGP4 positions of a catalogue object at UTC Julian dates, in GCRS (via astropy's TEME frame)."""
    import astropy.units as u
    from astropy.coordinates import GCRS, TEME, CartesianDifferential, CartesianRepresentation
    from astropy.time import Time

    sat = satrec_from_omm(omm)
    jd = np.floor(t_jd - 0.5) + 0.5
    fr = t_jd - jd
    err, r, v = sat.sgp4_array(jd, fr)
    if np.any(err != 0):
        raise ValueError(f"SGP4 failed for {omm.get('OBJECT_NAME')} (error {int(err[err != 0][0])}).")
    t = Time(t_jd, format="jd", scale="utc")
    rep = CartesianRepresentation(r.T * u.km, differentials=CartesianDifferential(v.T * u.km / u.s))
    g = TEME(rep, obstime=t).transform_to(GCRS(obstime=t))
    return Trajectory(np.asarray(t_jd), g.cartesian.xyz.to_value(u.km).T, g.velocity.d_xyz.to_value(u.km / u.s).T)


def interpolate(tr: Trajectory, t_jd: np.ndarray) -> Trajectory:
    """Positions between ephemeris points by cubic Hermite interpolation (positions and velocities both given)."""
    from scipy.interpolate import CubicHermiteSpline
    s = (tr.t_jd - tr.t_jd[0]) * 86400.0
    q = (np.asarray(t_jd) - tr.t_jd[0]) * 86400.0
    r = np.column_stack([CubicHermiteSpline(s, tr.r[:, i], tr.v[:, i])(q) for i in range(3)])
    v = np.column_stack([CubicHermiteSpline(s, tr.r[:, i], tr.v[:, i]).derivative()(q) for i in range(3)])
    return Trajectory(np.asarray(t_jd), r, v)


def ric(ref: Trajectory, other: Trajectory) -> dict[str, np.ndarray]:
    """Other minus reference in the reference's radial, in-track and cross-track directions (km), and the in-track
    offset as a time (s)."""
    R = ref.r / np.linalg.norm(ref.r, axis=1)[:, None]
    C = np.cross(ref.r, ref.v)
    C /= np.linalg.norm(C, axis=1)[:, None]
    I = np.cross(C, R)
    d = other.r - ref.r
    intrack = np.einsum("ij,ij->i", d, I)
    return {"radial": np.einsum("ij,ij->i", d, R), "intrack": intrack, "crosstrack": np.einsum("ij,ij->i", d, C),
            "intrack_s": intrack / np.linalg.norm(ref.v, axis=1)}


# --- observing -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    lat: float
    lon: float
    h_m: float = 0.0
    station: str = "9999"  # an IOD station number (assigned by the community); 9999 is a placeholder


def site_gcrs(site: Site, t_jd: np.ndarray) -> np.ndarray:
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    loc = EarthLocation.from_geodetic(site.lon * u.deg, site.lat * u.deg, site.h_m * u.m)
    pos, _ = loc.get_gcrs_posvel(Time(t_jd, format="jd", scale="utc"))
    return pos.xyz.to_value(u.km).T


C_KM_S = 299792.458


def light_time_jd(r_sat: np.ndarray, r_site: np.ndarray) -> np.ndarray:
    """How long the light took from the satellite to the site, in days (2 to 7 ms for a low orbit: 15 to 50 m along
    the track). Positions measured against catalogue stars are astrometric: the object where it was when the light
    left it, seen from where the site is when it arrived."""
    return np.linalg.norm(r_sat - r_site, axis=1) / C_KM_S / 86400.0


def radec(r_sat: np.ndarray, r_site: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topocentric RA, Dec (degrees) and elevation proxy (degrees above the plane perpendicular to the site's
    geocentric direction: within a fraction of a degree of the true elevation)."""
    d = r_sat - r_site
    n = np.linalg.norm(d, axis=1)
    ra = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    dec = np.degrees(np.arcsin(d[:, 2] / n))
    up = r_site / np.linalg.norm(r_site, axis=1)[:, None]
    el = np.degrees(np.arcsin(np.einsum("ij,ij->i", d, up) / n))
    return ra, dec, el


def angular_sep(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation, arcseconds (haversine: accurate at small angles)."""
    r1, d1, r2, d2 = (np.radians(np.asarray(x, float)) for x in (ra1, dec1, ra2, dec2))
    h = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600


def sunlit(r_sat: np.ndarray, t_jd: np.ndarray) -> np.ndarray:
    """Outside the Earth's cylindrical shadow (good to a few seconds at the shadow's edge)."""
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.time import Time
    s = get_sun(Time(t_jd, format="jd", scale="utc")).cartesian.xyz.to_value(u.km).T
    sh = s / np.linalg.norm(s, axis=1)[:, None]
    along = np.einsum("ij,ij->i", r_sat, sh)
    perp = np.linalg.norm(r_sat - along[:, None] * sh, axis=1)
    return (along > 0) | (perp > R_EARTH)


def dark_sky(r_site: np.ndarray, t_jd: np.ndarray, sun_below_deg: float = -6.0) -> np.ndarray:
    """The Sun below the given altitude at the site (civil twilight by default): when a camera can see satellites."""
    import astropy.units as u
    from astropy.coordinates import get_sun
    from astropy.time import Time
    s = get_sun(Time(t_jd, format="jd", scale="utc")).cartesian.xyz.to_value(u.km).T
    up = r_site / np.linalg.norm(r_site, axis=1)[:, None]
    toward = s - r_site
    alt = np.degrees(np.arcsin(np.einsum("ij,ij->i", toward, up) / np.linalg.norm(toward, axis=1)))
    return alt < sun_below_deg


def passes(t_jd: np.ndarray, el: np.ndarray, min_el: float = 15.0, min_points: int = 5) -> list[np.ndarray]:
    """Index runs where the object is above min_el: one per pass."""
    up = el >= min_el
    out, start = [], None
    for i, u_ in enumerate(up):
        if u_ and start is None:
            start = i
        if (not u_ or i == len(up) - 1) and start is not None:
            stop = i + 1 if u_ else i
            if stop - start >= min_points:
                out.append(np.arange(start, stop))
            start = None
    return out


# --- identification and the catalogue check ------------------------------------------------------------------------------

def identify(t_jd: np.ndarray, ra: np.ndarray, dec: np.ndarray, site: Site, catalogue: list[dict[str, Any]],
             max_clock_s: float = 5.0, top: int = 3) -> list[dict[str, Any]]:
    """Rank catalogue objects by how well they explain the observed track, allowing one clock offset (up to
    max_clock_s) since an observer's clock error shifts the object along its track."""
    rs = site_gcrs(site, t_jd)
    ranked = []
    for omm in catalogue:
        try:
            best = None
            for dt in np.linspace(-max_clock_s, max_clock_s, 21):
                tr = propagate_gp(omm, t_jd + dt / 86400.0)
                tr = propagate_gp(omm, t_jd + dt / 86400.0 - light_time_jd(tr.r, rs))
                cra, cdec, el = radec(tr.r, rs)
                if np.min(el) < -5:
                    best = None
                    break
                rms = float(np.sqrt(np.mean(angular_sep(ra, dec, cra, cdec) ** 2)))
                if best is None or rms < best[0]:
                    best = (rms, dt)
        except ValueError:
            continue
        if best is not None:
            ranked.append({"name": omm["OBJECT_NAME"], "norad": int(omm["NORAD_CAT_ID"]), "intl": omm["OBJECT_ID"],
                           "rms_arcsec": best[0], "clock_offset_s": best[1]})
    ranked.sort(key=lambda x: (round(x["rms_arcsec"], 3), x["norad"]))
    # Docked modules are catalogued separately with the station's own elements: one object, one answer.
    keys = {int(o["NORAD_CAT_ID"]): tuple(o.get(k) for k in ("EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
                                                             "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY"))
            for o in catalogue}
    grouped: list[dict[str, Any]] = []
    for x in ranked:
        twin = next((g for g in grouped if keys[g["norad"]] == keys[x["norad"]]), None)
        if twin is None:
            grouped.append({**x, "same_orbit": []})
        else:
            twin["same_orbit"].append(x["name"])
    return grouped[:top]


def catalogue_check(t_jd: np.ndarray, ra: np.ndarray, dec: np.ndarray, sigma_arcsec: float, site: Site,
                    omm: dict[str, Any]) -> dict[str, Any]:
    """One pass against the catalogue orbit: the time offset (in-track) and the cross-track offset that best
    explain the observations, with one-sigma errors, in seconds and km."""
    from scipy.optimize import least_squares
    rs = site_gcrs(site, t_jd)

    def model(p):
        dt, dc = p
        tr = propagate_gp(omm, t_jd + dt / 86400.0)
        tr = propagate_gp(omm, t_jd + dt / 86400.0 - light_time_jd(tr.r, rs))
        C = np.cross(tr.r, tr.v)
        C /= np.linalg.norm(C, axis=1)[:, None]
        cra, cdec, _ = radec(tr.r + dc * C, rs)
        return cra, cdec

    def resid(p):
        cra, cdec = model(p)
        dra = ((ra - cra + 180) % 360 - 180) * np.cos(np.radians(dec)) * 3600
        return np.concatenate([dra, (dec - cdec) * 3600]) / sigma_arcsec

    def jac(p):
        # Fixed absolute steps (10 ms, 10 m): a relative step is zero at the natural start, zero offset.
        base, cols = resid(p), []
        for i, h in enumerate((0.01, 0.01)):
            q = np.array(p, float)
            q[i] += h
            cols.append((resid(q) - base) / h)
        return np.column_stack(cols)

    sol = least_squares(resid, [0.0, 0.0], jac=jac)
    J = sol.jac
    cov = np.linalg.pinv(J.T @ J) * max(1.0, 2 * sol.cost / max(len(sol.fun) - 2, 1))
    v = float(np.linalg.norm(propagate_gp(omm, t_jd[:1]).v[0]))
    dt, dc = sol.x
    return {"time_offset_s": float(dt), "time_offset_sigma_s": float(math.sqrt(cov[0, 0])),
            "intrack_km": float(dt * v), "intrack_sigma_km": float(math.sqrt(cov[0, 0]) * v),
            "crosstrack_km": float(dc), "crosstrack_sigma_km": float(math.sqrt(cov[1, 1])),
            "speed_km_s": v, "km_per_clock_second": v,
            "residual_rms_arcsec": float(np.sqrt(np.mean((sol.fun * sigma_arcsec) ** 2)))}


# --- orbits with and without a manoeuvre ---------------------------------------------------------------------------------------------

def zonal_accel(r: np.ndarray) -> np.ndarray:
    """Acceleration from J2, J3 and J4 (km/s^2), with r in a frame whose z axis is the Earth's true pole: the gradient
    of the potential -(mu/r) sum_n Jn (Re/r)^n Pn(z/r) (Vallado, Fundamentals of Astrodynamics, the zonal terms)."""
    x, y, z = r
    r2 = x * x + y * y + z * z
    rn = math.sqrt(r2)
    s = z / rn
    m = MU_EARTH
    k2 = 1.5 * J2 * m * R_EARTH ** 2 / rn ** 5
    a2 = k2 * np.array([x * (5 * s * s - 1), y * (5 * s * s - 1), z * (5 * s * s - 3)])
    k3 = 2.5 * J3 * m * R_EARTH ** 3 / rn ** 7
    a3 = k3 * np.array([x * (7 * z ** 3 / r2 - 3 * z), y * (7 * z ** 3 / r2 - 3 * z), 7 * z ** 4 / r2 - 6 * z * z + 0.6 * r2])
    k4 = 15 / 8 * J4 * m * R_EARTH ** 4 / rn ** 7
    a4 = k4 * np.array([x * (1 - 14 * s * s + 21 * s ** 4), y * (1 - 14 * s * s + 21 * s ** 4),
                        z * (5 - 70 / 3 * s * s + 21 * s ** 4)])
    return a2 + a3 + a4


def true_pole_rotation(t_jd: float) -> np.ndarray:
    """GCRS to the true equator and equinox of date (astropy's TETE): its z axis is the Earth's pole that day, about
    0.15 degrees from GCRS's in 2026 because of precession. Over a few days it barely moves, so one matrix serves."""
    import astropy.units as u
    from astropy.coordinates import GCRS, TETE, CartesianRepresentation
    from astropy.time import Time
    t = Time(t_jd, format="jd", scale="utc")
    eye = CartesianRepresentation(np.eye(3) * u.one)
    g = GCRS(eye, obstime=t, representation_type="cartesian").transform_to(TETE(obstime=t))
    return g.cartesian.xyz.value  # columns: the GCRS axes seen in TETE


_POLE = {"t": None, "R": np.eye(3)}


def _accel(r: np.ndarray, v: np.ndarray, k_drag: float, R: np.ndarray | None = None) -> np.ndarray:
    """Two-body + J2, J3, J4 about the true pole (R: GCRS to true-of-date) + an along-track deceleration
    proportional to speed squared (drag, with the atmosphere's density folded into k_drag)."""
    rn = math.sqrt(r @ r)
    a = -MU_EARTH * r / rn ** 3
    if R is None:
        a = a + zonal_accel(r)
    else:
        a = a + R.T @ zonal_accel(R @ r)
    sp = math.sqrt(v @ v)
    return a - k_drag * sp * v


def integrate(state0: np.ndarray, t0_jd: float, t_jd: np.ndarray, k_drag: float,
              impulse: tuple[float, np.ndarray] | None = None, site: np.ndarray | None = None) -> np.ndarray:
    """Positions (km) at t_jd (all after t0_jd) from a state at t0_jd; impulse = (time jd, delta-v in radial,
    in-track, cross-track km/s) applied at that instant. With site (the observer's positions at t_jd), the positions
    are astrometric: each where the satellite was when its light left (the same solution, a few ms earlier)."""
    from scipy.integrate import solve_ivp

    if _POLE["t"] is None or abs(_POLE["t"] - t0_jd) > 1.0:  # one rotation per day of epochs is ample
        _POLE.update(t=t0_jd, R=true_pole_rotation(t0_jd))
    pole = _POLE["R"]  # (not R: the impulse below names its radial direction R)

    def rhs(_t, s):
        return np.concatenate([s[3:], _accel(s[:3], s[3:], k_drag, pole)])

    def leg(s0: np.ndarray, ta: float, tb: float):
        return solve_ivp(rhs, (ta, max(tb, ta + 1.0)), s0, rtol=1e-10, atol=1e-9, method="DOP853", dense_output=True)

    secs = (np.asarray(t_jd) - t0_jd) * 86400.0
    end = float(secs.max()) if len(secs) else 1.0
    if impulse is None:
        sol = leg(state0, 0.0, end).sol

        def at(x: np.ndarray) -> np.ndarray:
            return sol(x)[:3].T
    else:
        tm = (impulse[0] - t0_jd) * 86400.0
        first = leg(state0, 0.0, tm)
        s_m = first.sol(tm).copy()
        R = s_m[:3] / np.linalg.norm(s_m[:3])
        C = np.cross(s_m[:3], s_m[3:])
        C /= np.linalg.norm(C)
        I = np.cross(C, R)
        dv = impulse[1]
        s_m[3:] += dv[0] * R + dv[1] * I + dv[2] * C
        second = leg(s_m, tm, end)

        def at(x: np.ndarray) -> np.ndarray:
            out = np.zeros((len(x), 3))
            before = x < tm
            if before.any():
                out[before] = first.sol(x[before])[:3].T
            if (~before).any():
                out[~before] = second.sol(x[~before])[:3].T
            return out

    r = at(secs)
    if site is not None:
        # A few ms before the first point the dense solution extends its first step: metres at most.
        r = at(secs - light_time_jd(r, site) * 86400.0)
    return r


@dataclass
class OrbitFit:
    name: str
    theta: np.ndarray
    k: int
    rss: float  # in units of the measurement sigma
    n: int
    bic: float
    impulse_jd: float | None = None
    heldout_chi2: float = float("nan")
    heldout_rms_arcsec: float = float("nan")

    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {"law": self.name, "k": self.k, "bic": self.bic, "heldout_chi2": self.heldout_chi2,
                             "heldout_rms_arcsec": self.heldout_rms_arcsec, "k_drag": float(self.theta[6])}
        if self.impulse_jd is not None:
            d.update({"impulse_jd": self.impulse_jd, "dv_ric_m_s": [float(x * 1000) for x in self.theta[7:10]],
                      "dv_m_s": float(np.linalg.norm(self.theta[7:10]) * 1000)})
        return d


def fit_orbit(t_jd: np.ndarray, ra: np.ndarray, dec: np.ndarray, rs: np.ndarray, sigma_arcsec: float,
              state0: np.ndarray, t0_jd: float, impulse_jd: float | None = None) -> OrbitFit:
    """Least squares on the observed angles: the epoch state (6), the drag term, and (with impulse_jd) a delta-v."""
    from scipy.optimize import least_squares

    def unpack(th):
        imp = (impulse_jd, th[7:10]) if impulse_jd is not None else None
        return th[:6], th[6], imp

    def resid(th):
        s0, kd, imp = unpack(th)
        r = integrate(s0, t0_jd, t_jd, kd, imp, site=rs)  # where it was when the light left it
        cra, cdec, _ = radec(r, rs)
        dra = ((ra - cra + 180) % 360 - 180) * np.cos(np.radians(dec)) * 3600
        return np.concatenate([dra, (dec - cdec) * 3600]) / sigma_arcsec

    x0 = np.concatenate([state0, [1e-9], np.zeros(3) if impulse_jd is not None else []])
    scale = np.concatenate([[1, 1, 1, 1e-3, 1e-3, 1e-3], [1e-9], [1e-4] * 3 if impulse_jd is not None else []])
    steps = np.concatenate([[1e-3] * 3, [1e-6] * 3, [1e-11], [1e-6] * 3 if impulse_jd is not None else []])

    def jac(th):
        # Fixed absolute steps (1 m, 1 mm/s, a small drag change, 1 mm/s of impulse): the impulse starts at zero,
        # where a relative step would be zero.
        base, cols = resid(th), []
        for i, h in enumerate(steps):
            q = th.copy()
            q[i] += h
            cols.append((resid(q) - base) / h)
        return np.column_stack(cols)

    sol = least_squares(resid, x0, jac=jac, x_scale=scale, max_nfev=40)
    n = len(sol.fun)
    k = len(x0)
    rss = float(sol.fun @ sol.fun)
    return OrbitFit("one orbit" if impulse_jd is None else "one orbit and an impulse", sol.x, k, rss, n,
                    rss + k * math.log(n), impulse_jd)


def forecast_chi2(fit: OrbitFit, t_jd, ra, dec, rs, sigma_arcsec, t0_jd) -> tuple[float, float]:
    imp = (fit.impulse_jd, fit.theta[7:10]) if fit.impulse_jd is not None else None
    r = integrate(fit.theta[:6], t0_jd, t_jd, fit.theta[6], imp, site=rs)
    cra, cdec, _ = radec(r, rs)
    sep = angular_sep(ra, dec, cra, cdec)
    return float(np.mean((sep / sigma_arcsec) ** 2) / 2), float(np.sqrt(np.mean(sep ** 2)))


def detect_manoeuvre(t_jd: np.ndarray, ra: np.ndarray, dec: np.ndarray, site_pos: np.ndarray, sigma_arcsec: float,
                     state0: np.ndarray, pass_ids: np.ndarray) -> dict[str, Any]:
    """One orbit against one orbit plus an impulse in each gap between passes. Every law is fitted on the earlier
    75% of the passes and must forecast the rest (chi-square per angle at most HELDOUT_CHI2, errors scaled by the
    best law's training scatter); then BIC on all passes decides, with DECISIVE_DELTA_BIC to call a manoeuvre."""
    t0 = float(t_jd[0])
    ids = np.unique(pass_ids)
    cut_id = ids[int(len(ids) * TRAIN_FRACTION)]
    tr = pass_ids < cut_id
    gaps = [(float(t_jd[pass_ids == a].max()) + float(t_jd[pass_ids == b].min())) / 2 for a, b in itertools.pairwise(ids)]
    candidates = [None, *gaps]
    fits_all, fits_train = [], []
    for g in candidates:
        fits_all.append(fit_orbit(t_jd, ra, dec, site_pos, sigma_arcsec, state0, t0, g))
        if g is None or g < float(t_jd[tr].max()):
            fits_train.append(fit_orbit(t_jd[tr], ra[tr], dec[tr], site_pos[tr], sigma_arcsec, state0, t0, g))
        else:
            fits_train.append(None)  # an impulse after the training passes cannot be fitted from them
    best_scale = math.sqrt(min(f.rss / max(f.n - f.k, 1) for f in fits_all))
    scale = max(1.0, best_scale)
    for fa, ft in zip(fits_all, fits_train, strict=True):
        if ft is not None:
            chi2, rms = forecast_chi2(ft, t_jd[~tr], ra[~tr], dec[~tr], site_pos[~tr], sigma_arcsec * scale, t0)
            fa.heldout_chi2, fa.heldout_rms_arcsec = chi2, rms
    one = fits_all[0]
    one_fails = np.isfinite(one.heldout_chi2) and one.heldout_chi2 > HELDOUT_CHI2

    def forecasts(f: OrbitFit) -> bool:
        # Within the limit every law must meet; or, when one orbit cannot forecast the later passes at all, at least
        # RELATIVE_FORECAST times better than it (the orbit model's own small omissions keep every law above the
        # limit then, and the verdict says so).
        if not np.isfinite(f.heldout_chi2):
            return False
        return f.heldout_chi2 <= HELDOUT_CHI2 or (one_fails and f.heldout_chi2 * RELATIVE_FORECAST <= one.heldout_chi2)

    with_imp = [f for f in fits_all[1:] if forecasts(f)]
    best_imp = min(fits_all[1:], key=lambda f: f.bic) if len(fits_all) > 1 else None
    passing_imp = min(with_imp, key=lambda f: f.bic) if with_imp else None
    out: dict[str, Any] = {"one_orbit": one.summary(), "best_impulse": best_imp.summary() if best_imp else None,
                           "error_scale": scale, "gaps_tested": len(gaps),
                           "impulses": [f.summary() for f in fits_all[1:]]}
    if (passing_imp is not None and one.bic - passing_imp.bic >= DECISIVE_DELTA_BIC
            and (not np.isfinite(one.heldout_chi2) or passing_imp.heldout_chi2 <= one.heldout_chi2)):
        caveat = ("" if passing_imp.heldout_chi2 <= HELDOUT_CHI2 else
                  f"; its forecast misses still exceed the measurement noise (chi-square {passing_imp.heldout_chi2:.1f} against "
                  f"{one.heldout_chi2:.0f} for one orbit), so the size of the impulse is approximate")
        out.update(manoeuvre=True, impulse=passing_imp.summary(),
                   reason=f"an impulse of {np.linalg.norm(passing_imp.theta[7:10]) * 1000:.2f} m/s between passes fits "
                          f"{one.bic - passing_imp.bic:.0f} BIC better and forecasts the later passes{caveat}")
    elif one_fails:
        # One orbit cannot forecast the later passes, yet no impulse that could be tested does: the change (if any)
        # lies after the training passes, or is not an impulse. Say so rather than "no manoeuvre".
        out.update(manoeuvre=None, reason=("unresolved: one orbit cannot forecast the later passes, and no impulse in a gap the "
                                          "earlier passes cover does either; a change after them needs more passes to test"))
    else:
        out.update(manoeuvre=False, reason="one orbit explains every pass: no manoeuvre")
    return out


# --- tumbling, and the report ------------------------------------------------------------------------------------

def tumble_period(t_s: np.ndarray, mag: np.ndarray, min_s: float = 1.0, max_s: float = 600.0) -> dict[str, Any]:
    """The rotation period from a brightness series: the strongest periodogram peak, and whether twice it fits
    better (a symmetric body shows two flashes per turn, so the strongest peak is often half the period)."""
    from astropy.timeseries import LombScargle
    freq = np.linspace(1 / max_s, 1 / min_s, 20000)
    ls = LombScargle(t_s, mag)
    power = ls.power(freq)
    f0 = float(freq[np.argmax(power)])
    p0 = 1 / f0
    fold = []
    for p in (p0, 2 * p0):
        ph = (t_s % p) / p
        bins = np.clip((ph * 20).astype(int), 0, 19)
        means = np.array([mag[bins == b].mean() if np.any(bins == b) else np.nan for b in range(20)])
        fold.append(float(np.nanmean((mag - means[bins]) ** 2)))
    # Twice the period is the turn when its fold is clearly tighter (the two flashes differ).
    period = 2 * p0 if fold[1] < 0.8 * fold[0] else p0
    # Uncertainty from the peak's width: about the period squared over the series length.
    span = float(t_s.max() - t_s.min())
    return {"period_s": period, "strongest_peak_s": p0, "alias_checked": True, "doubled": period != p0,
            "sigma_s": period * period / span / 4, "false_alarm": float(ls.false_alarm_probability(power.max()))}


def _mx(value: float) -> str:
    """The IOD uncertainty code M*10^(X-8) for a value, rounding the mantissa up so it never claims more."""
    if value <= 0:
        return "  "
    x = math.floor(math.log10(value)) + 8
    m = math.ceil(value / 10 ** (x - 8) - 1e-9)
    if m == 10:
        m, x = 1, x + 1
    return f"{m}{min(max(x, 0), 9)}"


def iod_line(norad: int, intl: str, site: Site, t_jd: float, ra_deg: float, dec_deg: float, time_sigma_s: float,
             pos_sigma_arcmin: float, condition: str = "G") -> str:
    """One observation in the IOD format, angle format 2 (RA HHMMmmm, Dec +DDMMmm, uncertainty in arcminutes) and
    epoch code 5 (J2000). Columns as the specification: 1-15 object, 17-20 station, 22 condition, 24-40 UTC date and
    time, 42-43 time uncertainty, 45 format, 46 epoch, 48-61 position, 63-64 position uncertainty."""
    from astropy.time import Time
    t = Time(t_jd, format="jd", scale="utc").datetime
    yy, launch = intl[2:4], intl[5:].ljust(6)[:6]  # "1998-067A" -> "98", "067A"
    obj = f"{norad:05d} {yy} {launch}"
    stamp = f"{t:%Y%m%d%H%M%S}{int(t.microsecond / 1000):03d}"
    # Round in the smallest unit first, so a carry (59.9996 minutes) reaches the minutes and hours.
    ra_k = round((ra_deg % 360) / 15 * 60 * 1000) % (24 * 60 * 1000)  # thousandths of a minute of time
    ra_s = f"{ra_k // 60000:02d}{ra_k // 1000 % 60:02d}{ra_k % 1000:03d}"
    sign = "+" if dec_deg >= 0 else "-"
    dec_k = round(abs(dec_deg) * 60 * 100)  # hundredths of an arcminute
    dec_s = f"{dec_k // 6000:02d}{dec_k // 100 % 60:02d}{dec_k % 100:02d}"
    line = f"{obj:<15} {site.station:>4} {condition} {stamp} {_mx(time_sigma_s)} 25 {ra_s}{sign}{dec_s} {_mx(pos_sigma_arcmin)}"
    return line
