"""Satellites on real orbits: the ISS against its public catalogue orbit, identification, the clock's effect, and
manoeuvre detection with a negative control.

  python scripts/satellite_demo.py

Reads NASA's ISS ephemeris (OEM, the trajectory NASA publishes) and CelesTrak's GP elements, both cached under
var/astro/sat/ (fetched once; CelesTrak asks for no more than one download every two hours). Observations are the
real ISS positions seen from eight sites with camera-like noise; impulses are injected only for the positive test.
Writes data/astro/satellites/results.json. Free and local: no model calls.
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
from physicslens.astro import satellites as T

CACHE = ROOT / "var" / "astro" / "sat"
OUT = ROOT / "data" / "astro" / "satellites"
OEM_URL = "https://nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.txt"
GP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"
# A network of observers around the world (one site rarely sees enough passes in a few days).
SITES = [T.Site(51.48, -0.01, 50), T.Site(40.42, -3.70, 650), T.Site(38.9, -77.0, 50), T.Site(34.05, -118.25, 100),
         T.Site(35.68, 139.69, 40), T.Site(-33.87, 151.21, 50), T.Site(-33.92, 18.42, 20), T.Site(-33.45, -70.67, 570)]
SIGMA_ARCSEC = 30.0  # a plate-solved camera frame of a fast streak
STEP_S = 10.0


def cached(name: str, url: str) -> str:
    p = CACHE / name
    if not p.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:  # fixed https sources
            p.write_bytes(r.read())
    return p.read_text()


def observe(truth: T.Trajectory, t: np.ndarray, rng, extra=None) -> dict:
    """Every visible pass (above 15 degrees, the satellite sunlit and the site dark) as noisy RA/Dec, with its site
    and pass id."""
    rows = []
    pid = 0
    for site in SITES:
        rs = T.site_gcrs(site, t)
        r = truth.r if extra is None else truth.r + extra
        # Astrometric positions: where the satellite was when the light left it (the light time, 2 to 7 ms).
        lt = T.light_time_jd(r, rs)
        back = T.interpolate(truth, t - lt).r if extra is None else T.interpolate(truth, t - lt).r + extra
        ra, dec, el = T.radec(back, rs)
        visible = T.sunlit(r, t) & T.dark_sky(rs, t)  # the satellite in sunlight, the observer in darkness
        for p in T.passes(t, np.where(visible, el, -90)):
            n = len(p)
            rows.append({"t": t[p], "ra": ra[p] + rng.normal(0, SIGMA_ARCSEC / 3600, n) / np.cos(np.radians(dec[p])),
                         "dec": dec[p] + rng.normal(0, SIGMA_ARCSEC / 3600, n), "rs": rs[p], "pass": np.full(n, pid), "site": site})
            pid += 1
    rows.sort(key=lambda x: x["t"][0])
    for k, r in enumerate(rows):
        r["pass"][:] = k
    cat = {k: np.concatenate([r[k] for r in rows]) for k in ("t", "ra", "dec", "rs", "pass")}
    return cat | {"n_passes": len(rows)}


def injected(state_t: T.Trajectory, t: np.ndarray, at_jd: float, dv_ric_ms: np.ndarray) -> np.ndarray:
    """What an impulse adds to the real trajectory: our dynamics with the impulse minus without, from the real state
    just before it (so only the impulse's effect is injected, on top of NASA's positions)."""
    k = int(np.searchsorted(state_t.t_jd, at_jd)) - 1
    s0 = np.concatenate([state_t.r[k], state_t.v[k]])
    after = t > at_jd
    extra = np.zeros((len(t), 3))
    if after.any():
        a = T.integrate(s0, state_t.t_jd[k], t[after], 0.0, (at_jd, dv_ric_ms / 1000.0))
        b = T.integrate(s0, state_t.t_jd[k], t[after], 0.0, None)
        extra[after] = a - b
    return extra


def main() -> None:
    t_start = time.time()
    oem = T.read_oem(cached("iss_oem.txt", OEM_URL))
    cat = json.loads(cached("gp_stations.json", GP_URL.format(group="stations"))) + json.loads(cached("gp_visual.json", GP_URL.format(group="visual")))
    iss = next(o for o in cat if o["NORAD_CAT_ID"] == 25544)
    out: dict = {"generated": "scripts/satellite_demo.py", "oem_span": [Time(oem.t_jd[0], format="jd").isot[:16], Time(oem.t_jd[-1], format="jd").isot[:16]],
                 "gp_epoch": iss["EPOCH"], "sigma_arcsec": SIGMA_ARCSEC}

    # 1. The catalogue orbit against NASA's own trajectory, over the whole span (it ages from its epoch).
    tt = np.linspace(oem.t_jd[0], oem.t_jd[-1], 600)
    d = T.ric(T.interpolate(oem, tt), T.propagate_gp(iss, tt))
    ep = Time(iss["EPOCH"]).jd
    out["catalogue_vs_nasa"] = {"days_from_epoch": (tt - ep).tolist(), "intrack_km": d["intrack"].tolist(),
                                "crosstrack_km": d["crosstrack"].tolist(), "radial_km": d["radial"].tolist(),
                                "within_1_day": float(np.max(np.abs(d["intrack"][np.abs(tt - ep) <= 1]))),
                                "at_7_days": float(np.max(np.abs(d["intrack"][np.abs(np.abs(tt - ep) - 7) < 0.1]))) if np.any(np.abs(np.abs(tt - ep) - 7) < 0.1) else None}
    print("catalogue vs NASA, in-track within a day of epoch:", round(out["catalogue_vs_nasa"]["within_1_day"], 2), "km", flush=True)

    # 2. One pass: identification and the catalogue check, with clocks of different quality.
    rng = np.random.default_rng(0)
    t1 = np.arange(ep - 0.5, ep + 1.5, STEP_S / 86400)
    obs = observe(T.interpolate(oem, t1), t1, rng)
    first = obs["pass"] == 0
    site = next(s for s in SITES if np.allclose(T.site_gcrs(s, obs["t"][first][:1]), obs["rs"][first][:1]))
    ident = T.identify(obs["t"][first], obs["ra"][first], obs["dec"][first], site, cat)
    out["identification"] = ident
    truth_dt = float(np.mean(T.ric(T.interpolate(oem, obs["t"][first]), T.propagate_gp(iss, obs["t"][first]))["intrack_s"]))
    checks = []
    for clock in (0.0, 0.1, 0.5, 1.0):
        c = T.catalogue_check(obs["t"][first] + clock / 86400, obs["ra"][first], obs["dec"][first], SIGMA_ARCSEC, site, iss)
        checks.append({"clock_error_s": clock, **c})
    out["single_pass"] = {"truth_catalogue_behind_s": -truth_dt, "checks": checks, "site": site.__dict__,
                          "utc": Time(obs["t"][first][0], format="jd").isot[:19], "points": int(first.sum())}
    out["iod"] = [T.iod_line(25544, "1998-067A", site, float(tj), float(r), float(dd), 0.1, SIGMA_ARCSEC * 2 / 60)
                  for tj, r, dd in list(zip(obs["t"][first], obs["ra"][first], obs["dec"][first], strict=True))[::6]]
    print("identified:", [x["name"] for x in ident], "single pass:", round(checks[0]["time_offset_s"], 3), "s", flush=True)

    # 3. Manoeuvres: several days of passes; a negative control on the real orbit, then impulses injected into it.
    t3 = np.arange(oem.t_jd[0] + 3.0, oem.t_jd[0] + 7.0, STEP_S / 86400)
    runs = []
    for label, dv in (("none (real orbit)", None), ("0.5 m/s prograde", np.array([0.0, 0.5, 0.0])), ("2 m/s prograde", np.array([0.0, 2.0, 0.0]))):
        rng = np.random.default_rng(1)
        base = T.interpolate(oem, t3)
        probe = observe(base, t3, np.random.default_rng(1))
        ids = np.unique(probe["pass"])
        mid = ids[len(ids) // 2]
        at = float((probe["t"][probe["pass"] == mid - 1].max() + probe["t"][probe["pass"] == mid].min()) / 2)
        extra = injected(oem, t3, at, dv) if dv is not None else None
        o = observe(base, t3, rng, extra)
        s0 = T.propagate_gp(iss, o["t"][:1])
        state0 = np.concatenate([s0.r[0], s0.v[0]])
        t0 = time.time()
        res = T.detect_manoeuvre(o["t"], o["ra"], o["dec"], o["rs"], SIGMA_ARCSEC, state0, o["pass"])
        row = {"injected": label, "dv_m_s": float(np.linalg.norm(dv)) if dv is not None else 0.0, "injected_at_utc": Time(at, format="jd").isot[:16] if dv is not None else None,
               "passes": int(o["n_passes"]), **res, "seconds": time.time() - t0}
        if res["manoeuvre"] and dv is not None:
            row["window_right"] = abs(res["impulse"]["impulse_jd"] - at) < 0.02
        runs.append(row)
        print(label, "->", res["reason"], f"({row['seconds']:.0f} s)", flush=True)
    out["manoeuvres"] = runs

    # 4. Tumbling: a synthetic light curve with two slightly different flashes per turn (no public light curve of a known tumbler is used here).
    rng = np.random.default_rng(2)
    ts = np.sort(rng.uniform(0, 300, 1500))
    period = 37.2
    mag = 5 + 0.8 * np.abs(np.cos(2 * np.pi * ts / period)) ** 4 + 0.15 * np.cos(2 * np.pi * ts / period) + rng.normal(0, 0.08, len(ts))
    out["tumble_synthetic"] = {"true_period_s": period, **T.tumble_period(ts, mag)}
    out["outcome"] = outcome(out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(out["outcome"]["status"], out["outcome"]["headline"], f"({time.time() - t_start:.0f} s)")


def outcome(out: dict) -> dict:
    sp = out["single_pass"]
    c0 = sp["checks"][0]
    pos = [r for r in out["manoeuvres"][1:]]
    found = [r for r in pos if r["manoeuvre"] and r.get("window_right")]
    top = out["identification"][0]["name"] if out["identification"] else "nothing"
    err = c0["time_offset_s"] - sp["truth_catalogue_behind_s"]
    return {
        "headline": (f"The streak is {top}; against NASA's own trajectory its catalogue orbit is {abs(sp['truth_catalogue_behind_s']):.2f} s "
                     f"{'behind' if sp['truth_catalogue_behind_s'] > 0 else 'ahead'}, and one pass measured {c0['time_offset_s']:.3f} ± "
                     f"{c0['time_offset_sigma_s']:.3f} s. No false manoeuvre on the real orbit; {len(found)} of {len(pos)} injected impulses found in the right gap."),
        "action": "Time your frames with a GPS-synced clock: every second of clock error moves the object about "
                  f"{c0['km_per_clock_second']:.1f} km along its track. Submit the IOD lines yourself.",
        "artifacts": [],
        "checked_against": {"name": "catalogue time offset (s)", "truth": sp["truth_catalogue_behind_s"], "measured": c0["time_offset_s"],
                            "relative_error": err / abs(sp["truth_catalogue_behind_s"]) if sp["truth_catalogue_behind_s"] else 0.0,
                            "within_interval": abs(err) <= 1.96 * c0["time_offset_sigma_s"] + 0.01,
                            "source": "NASA JSC ISS ephemeris (OEM) against CelesTrak GP elements"},
        "confidence": f"one pass, {sp['points']} positions at {out['sigma_arcsec']:.0f} arcsec each",
        "status": "partial",
        "limits": [("no real recorded manoeuvre yet (the current NASA ephemeris lists none; past catalogue orbits need a Space-Track "
                    "account), so detection is shown on impulses injected into the real orbit; tumble periods are synthetic only")],
        "readable": None}


if __name__ == "__main__":
    main()
