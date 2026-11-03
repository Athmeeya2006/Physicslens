"""Residuals against JPL: the quality check of an observing setup, and how close our short-arc forecast came to JPL.

  python scripts/asteroid_jpl_check.py

For 2024 YR4 (the follow-up case of scripts/asteroid_demo.py), JPL Horizons gives the astrometric position seen from
each observatory at each observation time, from JPL's orbit fitted to every observation since. Two numbers come out:
- the stand-in night's measured positions minus JPL's (arcsec): what a user's residuals against JPL look like for a
  well-run setup, and the first thing to check about one's own;
- our forecast from the first nights plus that night minus JPL's position, over the later check window: how close a
  three-night orbit came to the orbit the whole world's observations later gave.
Free: MPC and Horizons public APIs, cached under var/astro/ast/. Writes data/astro/asteroids/jpl_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import asteroid_demo as D
from astropy.time import Time
from physicslens.astro import asteroids as A

OUT = D.OUT / "jpl_check.json"
CHUNK = 40  # epochs per Horizons request: its URL length limit


def jpl_radec(station: str, t_utc_jd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """JPL's astrometric RA and Dec (degrees, ICRF) of the object from an MPC site at UTC times."""
    def get():
        from astroquery.jplhorizons import Horizons
        ra, dec = [], []
        for i in range(0, len(t_utc_jd), CHUNK):
            # Observer tables read a list of Julian days as UT (Horizons' TLIST default for observer tables).
            e = Horizons(id=D.DESIG, id_type="smallbody", location=station,
                         epochs=[float(t) for t in t_utc_jd[i:i + CHUNK]]).ephemerides(quantities="1", extra_precision=True)
            ra += [float(v) for v in e["RA"]]
            dec += [float(v) for v in e["DEC"]]
        return {"ra": ra, "dec": dec}
    key = f"jpl_{station}_{len(t_utc_jd)}_{t_utc_jd.min():.5f}_{t_utc_jd.max():.5f}.json"
    d = D.cached_json(key, get)
    return np.array(d["ra"]), np.array(d["dec"])


def sky_offset(ra1, dec1, ra2, dec2) -> tuple[np.ndarray, np.ndarray]:
    """First minus second, arcsec on the sky (RA times cos dec)."""
    dra = ((np.asarray(ra1) - np.asarray(ra2) + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec2)) * 3600.0
    return dra, (np.asarray(dec1) - np.asarray(dec2)) * 3600.0


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2 + b ** 2) / 2))


def main() -> None:
    codes = A.parse_obscodes((D.CACHE / "ObsCodes.html").read_text(errors="replace"))
    rows = D.mpc_obs()
    obs = A.from_ades_rows(rows, codes)
    day = np.array([Time(t, format="jd", scale="utc").isot[:10] for t in obs.t_utc_jd])
    first = np.isin(day, D.FIRST_NIGHTS)
    yours = (day == D.YOUR_NIGHT[0]) & (obs.stn == D.YOUR_NIGHT[1])
    check = (day >= D.CHECK_FROM) & (day <= D.CHECK_TO)
    epoch = float(A.tdb(obs.t_utc_jd[first]).mean())
    bodies = A.planets(obs.t_utc_jd.min() - 1, obs.t_utc_jd.max() + D.AHEAD_DAYS + 2)
    before = A.fit(obs.subset(first), np.array(D.start_state(epoch)), epoch, bodies)
    after = A.fit(obs.subset(first | yours), before.state, epoch, bodies)

    # 1. The stand-in night against JPL: the setup check a user runs on their own night.
    sub_y = obs.subset(yours)
    jra, jdec = jpl_radec(D.YOUR_NIGHT[1], sub_y.t_utc_jd)
    ya, yb = sky_offset(sub_y.ra, sub_y.dec, jra, jdec)
    stated = float(np.sqrt(np.mean(sub_y.sigma_ra ** 2 + sub_y.sigma_dec ** 2) / 2))

    # 2. Our short-arc forecast against JPL's full-arc positions, at every later observation (per site).
    sub_c = obs.subset(check)
    ra_f, dec_f = A.predict(after.state, after.epoch_tdb, sub_c, bodies)
    cra, cdec = np.empty(len(sub_c.t_utc_jd)), np.empty(len(sub_c.t_utc_jd))
    for stn in sorted(set(sub_c.stn)):
        m = sub_c.stn == stn
        cra[m], cdec[m] = jpl_radec(str(stn), sub_c.t_utc_jd[m])
    fa, fb = sky_offset(ra_f, dec_f, cra, cdec)
    forecast_vs_jpl = np.hypot(fa, fb)
    # The later observations against JPL: how far real measurements sit from the best orbit (the noise floor).
    oa, ob = sky_offset(sub_c.ra, sub_c.dec, cra, cdec)
    obs_vs_jpl = np.hypot(oa, ob)

    out = {
        "generated": "scripts/asteroid_jpl_check.py", "object": D.DESIG,
        "reference": "JPL Horizons astrometric RA/Dec from each MPC site at each observation time (JPL's orbit from all observations)",
        "your_night": {"station": D.YOUR_NIGHT[1], "date": D.YOUR_NIGHT[0], "n": len(ya),
                       "residual_rms_arcsec": rms(ya, yb), "mean_ra_arcsec": float(np.mean(ya)),
                       "mean_dec_arcsec": float(np.mean(yb)), "stated_sigma_arcsec": stated,
                       "ra": ya.tolist(), "dec": yb.tolist()},
        "forecast_vs_jpl": {"window": [D.CHECK_FROM, D.CHECK_TO], "n": len(forecast_vs_jpl),
                            "from": f"orbit fitted to {D.FIRST_NIGHTS[0]}, {D.FIRST_NIGHTS[1]} and {D.YOUR_NIGHT[0]} ({D.YOUR_NIGHT[1]})",
                            "median_arcsec": float(np.median(forecast_vs_jpl)), "max_arcsec": float(np.max(forecast_vs_jpl))},
        "observations_vs_jpl": {"median_arcsec": float(np.median(obs_vs_jpl)), "max_arcsec": float(np.max(obs_vs_jpl))},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "your_night"} | {"your_night_rms": out["your_night"]["residual_rms_arcsec"],
                                                                          "stated": stated}, indent=1))


if __name__ == "__main__":
    main()
