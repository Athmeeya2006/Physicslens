"""The Sun's rotation law from real images, and when active regions face the Earth again.

Real pixels with a known answer: SDO/HMI continuum images (from Helioviewer, with the instrument's own header:
the observer's distance and Carrington position, the plate scale) show sunspots; their heliographic positions come
from sunpy using that header (the observer is never assumed). Laws for the sidereal rotation rate against latitude
phi are fitted to the spots' Carrington longitudes and must forecast the last quarter of each period:

  rigid         omega = A
  sin^2         omega = A + B sin^2 phi
  sin^2 sin^4   omega = A + B sin^2 phi + C sin^4 phi
  |phi|         omega = A + B |phi|             (a rival shape, not the textbook one)

Carrington longitude turns at 14.1844 deg/day by definition, so a spot's sidereal rate is 14.1844 plus its
Carrington drift: exact. The synodic rate (what an Earth observer sees) is the drift in Stonyhurst longitude; the
two differ by the Earth's orbital motion, measured here from the sub-Earth Carrington longitude at the dates (it
varies by about 3% over the year), not a fixed 0.9856 deg/day. Each law is linear in its coefficients and one
starting longitude per spot, so fitting is exact linear least squares.

Spots are not tracers of the plasma: their proper motions (a few hundredths of a degree per day, more for young
spots) are scatter, estimated from the data and stated.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CARRINGTON_DEG_PER_DAY = 14.1844  # sidereal rate of the Carrington frame, by definition
HELIOVIEWER = "https://api.helioviewer.org/v2"
HMI_CONTINUUM_SOURCE = 18
WORK_SIZE = 1024  # images are reduced to this many pixels across: 2.4 arcsec, about 1,700 km, per pixel
SPOT_THRESHOLD = 0.85  # darker than this fraction of the limb-darkened disc: umbra and penumbra
MIN_SPOT_PIXELS = 4
MAX_CENTRE_ANGLE_DEG = 65.0  # positions near the limb are foreshortened and biased: not used for rates
TRAIN_FRACTION = 0.75
HELDOUT_CHI2 = 4.0
DECISIVE_DELTA_BIC = 10.0
DAY = 86400.0


# --- images ------------------------------------------------------------------------------------------------------

def _get(url: str, timeout: float = 120.0) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:  # a fixed https host
        return r.read()


def parse_header(xml: str) -> dict[str, Any]:
    """The FITS keywords Helioviewer keeps in the JPEG 2000 file's XML box, as a dict (numbers converted)."""
    body = xml.split("<fits>", 1)[-1].split("</fits>", 1)[0]
    out: dict[str, Any] = {}
    for key, val in re.findall(r"<([A-Za-z0-9_\-]+)>([^<]*)</\1>", body):
        if key.lower() in ("history", "comment"):
            continue
        v = val.strip()
        try:
            out[key.lower()] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
        except ValueError:
            out[key.lower()] = v
    return out


def fetch_frame(when_utc: str, cache: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """The HMI continuum image closest to a time, reduced to WORK_SIZE, and its header (cached on disk)."""
    import io

    from PIL import Image

    cache.mkdir(parents=True, exist_ok=True)
    meta = json.loads(_get(f"{HELIOVIEWER}/getClosestImage/?date={when_utc}Z&sourceId={HMI_CONTINUUM_SOURCE}"))
    key = str(meta["id"])
    npz, hdr = cache / f"{key}.npz", cache / f"{key}.json"
    if npz.is_file() and hdr.is_file():
        return np.load(npz)["data"], json.loads(hdr.read_text())
    header = parse_header(_get(f"{HELIOVIEWER}/getJP2Header/?id={key}").decode("utf-8", "replace"))
    im = Image.open(io.BytesIO(_get(f"{HELIOVIEWER}/getJP2Image/?id={key}")))
    factor = int(header["naxis1"]) // WORK_SIZE
    im.reduce = round(math.log2(factor))  # decode at reduced resolution straight from the wavelet levels
    data = np.asarray(im, dtype=np.float32)
    if data.shape[0] != WORK_SIZE:
        data = np.asarray(Image.fromarray(data).resize((WORK_SIZE, WORK_SIZE)), dtype=np.float32)
    data = np.flipud(data)  # image rows run top-down; FITS rows bottom-up
    header = {**header, "naxis1": WORK_SIZE, "naxis2": WORK_SIZE,
              "cdelt1": header["cdelt1"] * factor, "cdelt2": header["cdelt2"] * factor,
              "crpix1": (header["crpix1"] - 0.5) / factor + 0.5, "crpix2": (header["crpix2"] - 0.5) / factor + 0.5,
              "helioviewer_id": key}
    np.savez_compressed(npz, data=data)
    hdr.write_text(json.dumps(header))
    return data, header


def sunpy_map(data: np.ndarray, header: dict[str, Any]):
    """A sunpy Map from the frame; the observer comes from the header's Carrington position and distance."""
    import sunpy.map
    keep = {k: v for k, v in header.items() if k not in ("hgln_obs", "hglt_obs", "bitpix", "blank", "bzero", "bscale",
                                                         "checksum", "datasum", "helioviewer_id")}
    return sunpy.map.Map(data.astype(np.float64), keep)


# --- spots -------------------------------------------------------------------------------------------------------

@dataclass
class Spot:
    t: float  # days since the period's start
    x: float  # pixel
    y: float
    area_px: float
    lat: float  # Carrington latitude (deg)
    lon_carr: float  # Carrington longitude (deg)
    lon_stony: float  # Stonyhurst longitude (deg): 0 faces the observer
    mu: float  # cosine of the angle from disc centre


def disc_geometry(header: dict[str, Any]) -> tuple[float, float, float]:
    """Disc centre (x, y, 0-based pixels) and radius (pixels) from the WCS: the reference value is the disc centre."""
    cx = header["crpix1"] - 1 - header.get("crval1", 0.0) / header["cdelt1"]
    cy = header["crpix2"] - 1 - header.get("crval2", 0.0) / header["cdelt2"]
    return cx, cy, header["rsun_obs"] / header["cdelt1"]


def flatten_disc(data: np.ndarray, header: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The image divided by its limb-darkening profile (the median at each distance from centre), and mu."""
    cx, cy, r = disc_geometry(header)
    yy, xx = np.indices(data.shape)
    rho = np.hypot(xx - cx, yy - cy) / r
    on = rho < 0.995
    bins = np.clip((rho * 200).astype(int), 0, 200)
    med = np.zeros(201)
    for b in np.unique(bins[on]):
        med[b] = np.median(data[on & (bins == b)])
    good = med > 0
    profile = np.interp(np.arange(201), np.flatnonzero(good), med[good])
    flat = np.where(on, data / np.maximum(profile[bins], 1e-6), np.nan)
    return flat, np.sqrt(np.clip(1 - rho ** 2, 0, 1))


def detect_spots(data: np.ndarray, header: dict[str, Any], t_days: float) -> list[Spot]:
    """Dark features on the flattened disc, centroided by how dark they are, placed on the Sun with sunpy."""
    import astropy.units as u
    from scipy import ndimage
    from sunpy.coordinates import frames

    flat, mu = flatten_disc(data, header)
    dark = np.nan_to_num(flat, nan=1.0) < SPOT_THRESHOLD
    labels, n = ndimage.label(dark, structure=np.ones((3, 3)))
    if n == 0:
        return []
    weight = np.clip(1 - np.nan_to_num(flat, nan=1.0), 0, None)
    idx = np.arange(1, n + 1)
    area = ndimage.sum(dark, labels, idx)
    cy_, cx_ = np.array(ndimage.center_of_mass(weight, labels, idx)).T
    keep = area >= MIN_SPOT_PIXELS
    if not keep.any():
        return []
    m = sunpy_map(data, header)
    hp = m.pixel_to_world(cx_[keep] * u.pix, cy_[keep] * u.pix)
    carr = hp.transform_to(frames.HeliographicCarrington(observer=m.observer_coordinate, obstime=m.date))
    stony = hp.transform_to(frames.HeliographicStonyhurst(obstime=m.date))
    out = []
    for i, (x, y, a) in enumerate(zip(cx_[keep], cy_[keep], area[keep], strict=True)):
        lat, lc, ls = carr.lat[i].to_value(u.deg), carr.lon[i].to_value(u.deg), stony.lon[i].to_value(u.deg)
        if not (np.isfinite(lat) and np.isfinite(lc)):
            continue
        out.append(Spot(t_days, float(x), float(y), float(a), float(lat), float(lc) % 360, float(ls),
                        float(mu[int(np.rint(y)), int(np.rint(x))])))
    return out


# --- tracks ------------------------------------------------------------------------------------------------------

@dataclass
class Track:
    spots: list[Spot] = field(default_factory=list)

    @property
    def lat(self) -> float:
        return float(np.average([s.lat for s in self.spots], weights=[s.area_px for s in self.spots]))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.array([s.t for s in self.spots])
        lc = np.unwrap(np.radians([s.lon_carr for s in self.spots]))
        ls = np.unwrap(np.radians([s.lon_stony for s in self.spots]))
        return t, np.degrees(lc), np.degrees(ls)


def _dlon(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def link(frames: list[list[Spot]], max_dlat: float = 1.5, max_dlon: float = 2.5) -> list[Track]:
    """Spots barely move in Carrington coordinates (well under a degree per day against 13 in the sky), so each is
    joined to the nearest spot of the previous frame within a small box; the biggest spots claim first. A spot that
    splits or merges ends one track and starts another."""
    live: list[Track] = []
    done: list[Track] = []
    for spots in frames:
        claimed: set[int] = set()
        nxt: list[Track] = []
        for s in sorted(spots, key=lambda s: -s.area_px):
            best, bd = None, None
            for k, tr in enumerate(live):
                if k in claimed:
                    continue
                last = tr.spots[-1]
                dt = s.t - last.t
                dl = abs(_dlon(s.lon_carr, last.lon_carr))
                if abs(s.lat - last.lat) <= max_dlat and dl <= max_dlon + 0.5 * dt:
                    d = math.hypot(s.lat - last.lat, dl)
                    if bd is None or d < bd:
                        best, bd = k, d
            if best is None:
                nxt.append(Track([s]))
            else:
                claimed.add(best)
                live[best].spots.append(s)
                nxt.append(live[best])
        done += [tr for k, tr in enumerate(live) if k not in claimed]
        live = nxt
    return done + live


def usable(tracks: list[Track], min_points: int = 8, min_days: float = 2.0) -> list[Track]:
    """Tracks long enough to give a rate, with only the points away from the limb."""
    out = []
    for tr in tracks:
        pts = [s for s in tr.spots if abs(s.lon_stony) <= MAX_CENTRE_ANGLE_DEG and math.degrees(math.acos(min(1, s.mu))) <= MAX_CENTRE_ANGLE_DEG + 5]
        if len(pts) >= min_points and pts[-1].t - pts[0].t >= min_days:
            out.append(Track(pts))
    return out


def track_rates(tr: Track) -> dict[str, float]:
    """One spot's sidereal rate (Carrington drift plus the frame's rate), synodic rate (Stonyhurst drift), with
    their least-squares errors from the scatter about a straight line."""
    t, lc, ls = tr.arrays()
    out: dict[str, float] = {"lat": tr.lat, "n": len(t), "days": float(t[-1] - t[0])}
    for name, y in (("sidereal", lc), ("synodic", ls)):
        A = np.column_stack([np.ones_like(t), t])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        s2 = float(resid @ resid) / max(len(t) - 2, 1)
        cov = s2 * np.linalg.inv(A.T @ A)
        rate = coef[1] + (CARRINGTON_DEG_PER_DAY if name == "sidereal" else 0.0)
        out[name] = float(rate)
        out[f"{name}_sigma"] = float(math.sqrt(cov[1, 1]))
        out[f"{name}_scatter_deg"] = math.sqrt(s2)
    return out


# --- laws --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Law:
    id: str
    title: str
    latex: str
    basis: tuple[str, ...]  # functions of latitude multiplying the coefficients

    def design(self, lat_deg: np.ndarray) -> np.ndarray:
        s2 = np.sin(np.radians(lat_deg)) ** 2
        cols = {"1": np.ones_like(s2), "s2": s2, "s4": s2 * s2, "abs": np.abs(np.radians(lat_deg))}
        return np.column_stack([cols[b] for b in self.basis])


LAWS = {
    "rigid": Law("rigid", "Rigid rotation", r"\omega = A", ("1",)),
    "sin2": Law("sin2", "Differential rotation (sin² φ)", r"\omega = A + B\sin^2\varphi", ("1", "s2")),
    "sin2sin4": Law("sin2sin4", "Differential rotation (sin² φ and sin⁴ φ)", r"\omega = A + B\sin^2\varphi + C\sin^4\varphi",
                    ("1", "s2", "s4")),
    "abs": Law("abs", "Rate falling linearly with |φ|", r"\omega = A + B|\varphi|", ("1", "abs")),
}


@dataclass
class LawFit:
    law: Law
    coef: np.ndarray
    cov: np.ndarray
    rss: float
    n: int
    k: int
    heldout_rms_deg: float = float("nan")
    heldout_chi2: float = float("nan")
    bic: float = float("nan")

    def omega(self, lat_deg) -> np.ndarray:
        return self.law.design(np.atleast_1d(np.asarray(lat_deg, float))) @ self.coef

    def omega_sigma(self, lat_deg) -> np.ndarray:
        X = self.law.design(np.atleast_1d(np.asarray(lat_deg, float)))
        return np.sqrt(np.einsum("ij,jk,ik->i", X, self.cov, X))

    def summary(self) -> dict[str, Any]:
        names = "ABC"[: len(self.coef)]
        return {"law": self.law.id, "title": self.law.title, "latex": self.law.latex,
                "coef": {n: float(v) for n, v in zip(names, self.coef, strict=True)},
                "sigma": {n: float(math.sqrt(self.cov[i, i])) for i, n in enumerate(names)},
                "heldout_rms_deg": self.heldout_rms_deg, "heldout_chi2": self.heldout_chi2, "bic": self.bic, "n": self.n}


def _system(tracks: list[tuple[int, np.ndarray, np.ndarray, float]], law: Law, n_tracks: int):
    """Rows: Carrington longitude = L_k + (omega(lat_k) - 14.1844) t. Unknowns: the law's coefficients, then one
    starting longitude per track. Returns (A, y)."""
    rows, ys = [], []
    for k, t, lc, lat in tracks:
        X = law.design(np.full(len(t), lat)) * t[:, None]
        L = np.zeros((len(t), n_tracks))
        L[:, k] = 1.0
        rows.append(np.hstack([X, L]))
        ys.append(lc + CARRINGTON_DEG_PER_DAY * t)
    return np.vstack(rows), np.concatenate(ys)


def fit_law(law: Law, tracks: list[Track], t_cut: float | None = None) -> LawFit:
    """Fit on every point before t_cut (all when None); each track contributes its own starting longitude."""
    data = []
    for tr in tracks:
        t, lc, _ = tr.arrays()
        m = t < t_cut if t_cut is not None else np.ones(len(t), bool)
        if m.sum() >= 2:
            data.append((len(data), t[m], lc[m], tr.lat))
    A, y = _system(data, law, len(data))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    k = len(law.basis)
    dof = max(len(y) - A.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov_full = s2 * np.linalg.pinv(A.T @ A)
    return LawFit(law, coef[:k], cov_full[:k, :k], float(resid @ resid), len(y), A.shape[1])


def forecast_score(law: Law, tracks: list[Track], t_cut: float, sigma_deg: float) -> tuple[float, float, int]:
    """Fit before t_cut, predict every later point of the tracks seen before it: (rms deg, chi2 per point, n)."""
    errs = []
    fit = fit_law(law, tracks, t_cut)
    ki = 0
    for tr in tracks:
        t, lc, _ = tr.arrays()
        before, after = t < t_cut, t >= t_cut
        if before.sum() < 2:
            continue
        rate = float(fit.omega(tr.lat)[0]) - CARRINGTON_DEG_PER_DAY
        # This track's own starting longitude, from its training points only, at the law's rate.
        L = float(np.mean(lc[before] - rate * t[before]))
        ki += 1
        if after.any():
            errs.append(lc[after] - (L + rate * t[after]))
    if not errs:
        return float("nan"), float("nan"), 0
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean((e / sigma_deg) ** 2)), len(e)


def assess(periods: list[list[Track]]) -> dict[str, Any]:
    """Every law fitted on all periods together; each must forecast the last quarter of every period (positions in
    degrees, errors scaled by the best law's own training scatter), then BIC decides."""
    allt = [tr for p in periods for tr in p]
    fits = {lid: fit_law(law, allt) for lid, law in LAWS.items()}
    # The common error scale: the scatter the best-fitting law leaves in training (proper motions and measurement).
    best = min(fits.values(), key=lambda f: f.rss / max(f.n - f.k, 1))
    sigma = math.sqrt(best.rss / max(best.n - best.k, 1))
    for f in fits.values():
        e2, n = [], 0
        for p in periods:
            if not p:
                continue
            t0 = min(s.t for tr in p for s in tr.spots)
            t1 = max(s.t for tr in p for s in tr.spots)
            cut = t0 + TRAIN_FRACTION * (t1 - t0)
            rms, _, m = forecast_score(f.law, p, cut, sigma)
            if m:
                e2.append(rms ** 2 * m)
                n += m
        f.heldout_rms_deg = math.sqrt(sum(e2) / n) if n else float("nan")
        f.heldout_chi2 = (f.heldout_rms_deg / sigma) ** 2 if n else float("nan")
        f.bic = f.n * math.log(f.rss / f.n) + f.k * math.log(f.n)
    passing = sorted((f for f in fits.values() if f.heldout_chi2 <= HELDOUT_CHI2), key=lambda f: f.bic)
    if not passing:
        return {"status": "no_fit", "winner": None, "fits": fits, "sigma_deg": sigma,
                "reason": "No law forecasts the later spot positions within the scatter."}
    lead = passing[0]
    close = [f for f in passing[1:] if f.bic - lead.bic < DECISIVE_DELTA_BIC and len(f.coef) <= len(lead.coef)]
    if close:
        return {"status": "ambiguous", "winner": None, "contenders": [lead, *close], "fits": fits, "sigma_deg": sigma,
                "reason": f"{lead.law.title} and {close[0].law.title} forecast equally well (BIC within {DECISIVE_DELTA_BIC:.0f})."}
    runner = passing[1] if len(passing) > 1 else None
    return {"status": "accepted", "winner": lead, "fits": fits, "sigma_deg": sigma,
            "reason": f"{lead.law.title} forecasts the later positions (held-out chi-square {lead.heldout_chi2:.2f} per point)"
                      + (f"; next best ({runner.law.title}) {runner.bic - lead.bic:.0f} BIC worse" if runner else "") + "."}


# --- published profiles, Earth's motion, forecasts ---------------------------------------------------------------------

# Sidereal A, B, C (A + B sin^2 + C sin^4) as implemented in sunpy.sun.models.differential_rotation: Howard, Harvey &
# Forgach 1990 (Sol. Phys. 130, 295) and Snodgrass 1983 (ApJ 270, 288) in microradians per second; Allen's
# Astrophysical Quantities in degrees per day.
URAD_S_TO_DEG_DAY = math.degrees(1e-6) * DAY
PUBLISHED = {
    "howard": {"source": "Howard, Harvey & Forgach 1990, Sol. Phys. 130, 295", "ABC": [x * URAD_S_TO_DEG_DAY for x in (2.894, -0.428, -0.370)]},
    "snodgrass": {"source": "Snodgrass 1983, ApJ 270, 288", "ABC": [x * URAD_S_TO_DEG_DAY for x in (2.851, -0.343, -0.474)]},
    "allen": {"source": "Allen's Astrophysical Quantities", "ABC": [14.44, -3.0, 0.0]},
}


def published_omega(name: str, lat_deg) -> np.ndarray:
    a, b, c = PUBLISHED[name]["ABC"]
    s2 = np.sin(np.radians(np.asarray(lat_deg, float))) ** 2
    return a + b * s2 + c * s2 * s2


def earth_rate_deg_per_day(when_utc: str) -> float:
    """How fast the sub-Earth point moves in Carrington longitude, against the frame's own rate: the difference
    between sidereal and synodic rates on that date (sunpy's L0, a centred difference over a day)."""
    import astropy.units as u
    from astropy.time import Time
    from sunpy.coordinates import sun

    t = Time(when_utc)
    d = (sun.L0(t + 0.5 * u.day) - sun.L0(t - 0.5 * u.day)).to_value(u.deg)
    d = (d + 180) % 360 - 180  # L0 decreases by about 13.2 degrees a day
    return CARRINGTON_DEG_PER_DAY + d


def forecast_region(lat: float, lon_stony: float, t_utc: str, law: LawFit, earth_rate: float,
                    rate_scatter: float) -> dict[str, Any]:
    """From a region's position now: when it crosses the central meridian, leaves at the west limb, and comes back
    at the east limb, with one-sigma uncertainties from the law's interval and spots' own rate scatter."""
    from astropy.time import Time, TimeDelta

    syn = float(law.omega(lat)[0]) - earth_rate
    sig = math.hypot(float(law.omega_sigma(lat)[0]), rate_scatter)
    t0 = Time(t_utc)

    def when(deg_ahead: float) -> dict[str, Any]:
        dt = deg_ahead / syn
        return {"utc": (t0 + TimeDelta(dt * DAY, format="sec")).isot[:16], "days": dt, "sigma_days": abs(dt) * sig / syn}

    to_cm = (-lon_stony) % 360 if lon_stony > 0 else -lon_stony
    return {"lat": lat, "lon_stony": lon_stony, "synodic_deg_day": syn, "central_meridian": when(to_cm) if lon_stony < 0 else None,
            "west_limb": when(90 - lon_stony), "east_limb_return": when(270 - lon_stony)}


# --- NOAA's Solar Region Summaries -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    number: int
    date: str  # the summary's "valid at" date (UTC, YYYY-MM-DD)
    lat: float
    lon_stony: float  # east negative, west positive
    lon_carr: float
    area: int


def parse_srs(text: str, date: str) -> tuple[list[Region], list[dict[str, Any]]]:
    """Section I (regions with spots) and section II (regions due to return: number, latitude, Carrington lon)."""
    regions, returns = [], []
    section = None
    for ln in text.splitlines():
        if ln.startswith("I.  Regions with Sunspots"):
            section = "I"
            continue
        if ln.startswith("IA."):
            section = "IA"
            continue
        if ln.startswith("II."):
            section = "II"
            continue
        m = re.match(r"^\s*(\d{4,5})\s+([NS])(\d\d)([EW])(\d\d)\s+(\d{3})\s+(\d{4})", ln)
        if section == "I" and m:
            num, ns, la, ew, lo, carr, area = m.groups()
            regions.append(Region(int(num), date, (1 if ns == "N" else -1) * float(la), (1 if ew == "W" else -1) * float(lo),
                                  float(carr), int(area)))
            continue
        m2 = re.match(r"^\s*(\d{4,5})\s+([NS])(\d\d)\s+(\d{3})\s*$", ln)
        if section == "II" and m2:
            num, ns, la, carr = m2.groups()
            returns.append({"number": int(num), "lat": (1 if ns == "N" else -1) * float(la), "lon_carr": float(carr), "listed": date})
    return regions, returns
