"""Did my observation matter? Asteroid 2024 YR4's orbit from its first nights, one more night added, and the
forecasts checked against the observations that came later.

  python scripts/asteroid_demo.py

Reads the MPC's observations of 2024 YR4 (its public API, ADES fields with each observation's uncertainty) and the
MPC observatory list, cached under var/astro/ast/. JPL Horizons gives only the starting guess for the fit (the fit
then uses the observations alone). Writes data/astro/asteroids/results.json and an example ADES report. Free.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from astropy.time import Time
from physicslens.astro import asteroids as A

CACHE = ROOT / "var" / "astro" / "ast"
OUT = ROOT / "data" / "astro" / "asteroids"
DESIG = "2024 YR4"
FIRST_NIGHTS = ("2024-12-25", "2024-12-26")
YOUR_NIGHT = ("2024-12-27", "W68")  # one observatory's tracklet the next night (ATLAS Chile), standing in for yours
CHECK_FROM, CHECK_TO = "2025-01-17", "2025-01-24"  # later observations the forecasts must meet
AHEAD_DAYS = 30


def cached_json(name: str, fetch) -> object:
    p = CACHE / name
    if not p.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fetch()))
    return json.loads(p.read_text())


def mpc_obs() -> list[dict]:
    def get():
        req = urllib.request.Request("https://data.minorplanetcenter.net/api/get-obs", method="GET",
                                     data=json.dumps({"desigs": [DESIG], "output_format": ["ADES_DF"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())[0]["ADES_DF"]
    return cached_json("yr4_ades.json", get)


def start_state(epoch_tdb: float) -> list[float]:
    def get():
        from astroquery.jplhorizons import Horizons
        v = Horizons(id=DESIG, id_type="smallbody", location="@sun", epochs=epoch_tdb).vectors(refplane="earth")
        return [float(v[k][0]) for k in ("x", "y", "z", "vx", "vy", "vz")]
    return cached_json(f"yr4_start_{epoch_tdb:.5f}.json", get)


def main() -> None:
    t_start = time.time()
    codes_path = CACHE / "ObsCodes.html"
    if not codes_path.is_file():
        with urllib.request.urlopen("https://minorplanetcenter.net/iau/lists/ObsCodes.html", timeout=120) as r:
            codes_path.write_bytes(r.read())
    codes = A.parse_obscodes(codes_path.read_text(errors="replace"))
    rows = mpc_obs()
    obs = A.from_ades_rows(rows, codes)
    day = np.array([Time(t, format="jd", scale="utc").isot[:10] for t in obs.t_utc_jd])
    first = np.isin(day, FIRST_NIGHTS)
    yours = (day == YOUR_NIGHT[0]) & (obs.stn == YOUR_NIGHT[1])
    check = (day >= CHECK_FROM) & (day <= CHECK_TO)
    epoch = float(A.tdb(obs.t_utc_jd[first]).mean())
    bodies = A.planets(obs.t_utc_jd.min() - 1, obs.t_utc_jd.max() + AHEAD_DAYS + 2)
    s0 = np.array(start_state(epoch))

    before = A.fit(obs.subset(first), s0, epoch, bodies)
    after = A.fit(obs.subset(first | yours), before.state, epoch, bodies)
    t_last = float(obs.t_utc_jd[first | yours].max())
    t_ahead = t_last + AHEAD_DAYS
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric
    tt = Time(t_ahead, format="jd", scale="utc").tdb
    geo = (get_body_barycentric("earth", tt) - get_body_barycentric("sun", tt)).xyz.to_value(u.au)
    unc_b = A.position_uncertainty(before, t_ahead, geo, bodies)
    unc_a = A.position_uncertainty(after, t_ahead, geo, bodies)

    def forecast(sol: A.OrbitSolution) -> dict:
        sub = obs.subset(check)
        ra_c, dec_c = A.predict(sol.state, sol.epoch_tdb, sub, bodies)
        a, b = A.residuals(ra_c, dec_c, sub)
        miss = np.hypot(a, b)
        mid = float(sub.t_utc_jd.mean())
        tm = Time(mid, format="jd", scale="utc").tdb
        g = (get_body_barycentric("earth", tm) - get_body_barycentric("sun", tm)).xyz.to_value(u.au)
        unc = A.position_uncertainty(sol, mid, g, bodies)
        return {"n": len(miss), "median_miss_arcsec": float(np.median(miss)), "max_miss_arcsec": float(np.max(miss)),
                "predicted_1sigma_major_arcsec": unc["major_arcsec"],
                "within_3_sigma": float(np.mean(miss <= 3 * unc["major_arcsec"] + 3 * np.hypot(sub.sigma_ra, sub.sigma_dec)))}

    f_b, f_a = forecast(before), forecast(after)
    # Your night against the orbit fitted without it: the quality check of a setup (residuals in arcsec).
    sub_y = obs.subset(yours)
    ra_c, dec_c = A.predict(before.state, before.epoch_tdb, sub_y, bodies)
    ry = A.residuals(ra_c, dec_c, sub_y)
    your_rows = [r for r in rows if r["obstime"][:10] == YOUR_NIGHT[0] and r["stn"] == YOUR_NIGHT[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "example_ades.psv").write_text(A.ades_psv(sub_y, your_rows))
    out = {"generated": "scripts/asteroid_demo.py", "object": DESIG, "first_nights": list(FIRST_NIGHTS), "your_night": list(YOUR_NIGHT),
           "n_first": int(first.sum()), "n_yours": int(yours.sum()), "ahead_days": AHEAD_DAYS,
           "ahead_utc": Time(t_ahead, format="jd", scale="utc").isot[:10],
           "before": {"rms_arcsec": before.rms_arcsec, "chi2_per_dof": before.chi2_per_dof, "rejected": int((~before.used).sum()),
                      "uncertainty_ahead": unc_b, "forecast_check": f_b},
           "after": {"rms_arcsec": after.rms_arcsec, "chi2_per_dof": after.chi2_per_dof, "rejected": int((~after.used).sum()),
                     "uncertainty_ahead": unc_a, "forecast_check": f_a},
           "your_residuals_arcsec": {"ra": ry[0].tolist(), "dec": ry[1].tolist(), "rms": float(np.sqrt(np.mean(ry[0] ** 2 + ry[1] ** 2) / 2))},
           "check_window": [CHECK_FROM, CHECK_TO]}
    shrink = unc_b["major_arcsec"] / max(unc_a["major_arcsec"], 1e-9)
    honest = f_a["within_3_sigma"] >= 0.9
    out["outcome"] = {
        "headline": (f"One more night shrinks 2024 YR4's position uncertainty a month ahead from {unc_b['major_arcsec']:.1f} to "
                     f"{unc_a['major_arcsec']:.1f} arcsec ({shrink:.1f} times smaller); the forecasts met the observations made three "
                     f"weeks later within {f_a['median_miss_arcsec']:.1f} arcsec (median)."),
        "action": "Submit your night as ADES (example below) with your own MPC code; follow-up in the first days matters most.",
        "artifacts": [], "checked_against": {"name": "forecast miss three weeks later (arcsec, median)", "truth": 0.0,
                                             "measured": f_a["median_miss_arcsec"], "relative_error": 0.0,
                                             "within_interval": honest, "source": "MPC observations of 2024 YR4, 2025 January 17 to 24"},
        "confidence": f"{f_a['within_3_sigma']:.0%} of the later observations within three of the stated sigmas",
        "status": "proven" if honest and shrink > 1.2 else "partial",
        "limits": [] if honest else ["the stated uncertainty is too small for the later observations"], "readable": None}
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("before", "after", "your_residuals_arcsec")}, default=float)[:1500])
    print(out["outcome"]["status"], out["outcome"]["headline"], f"({time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
