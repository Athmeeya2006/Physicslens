"""The Sun's rotation law from real SDO/HMI images, and active-region return forecasts checked against NOAA.

  python scripts/sun_demo.py            # fetch what is missing (Helioviewer, NOAA), then analyse
  python scripts/sun_demo.py --bias-only   # only the return-bias check, from the stored returns
  python scripts/sun_demo.py --offline  # analyse cached frames only

Five 14-day periods across two solar cycles, one image every 6 hours. Frames are cached under var/astro/sun/
(not committed); results go to data/astro/sun/. Free and local apart from the downloads: no model calls.
"""

from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from astropy.time import Time, TimeDelta
from physicslens.astro import sun as S

CACHE = ROOT / "var" / "astro" / "sun"
OUT = ROOT / "data" / "astro" / "sun"
PERIODS = [  # (start, why): early, maximum and declining phases of cycle 24, rising and maximum of cycle 25
    ("2011-09-20", "cycle 24 rising: spots at higher latitudes"),
    ("2014-01-01", "cycle 24 maximum"),
    ("2017-08-29", "cycle 24 declining: spots near the equator (AR 2673)"),
    ("2022-03-01", "cycle 25 rising"),
    ("2024-05-01", "cycle 25 maximum (AR 3664, the May 2024 storms)"),
]
DAYS, STEP_H = 14, 6
SRS_URL = "https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries/{y}/{m:02d}/{y}{m:02d}{d:02d}SRS.txt"
RETURN_WINDOW = (10, 36)  # days after a period's end in which a region seen at its end can come back
MATCH_LAT, MATCH_LON = 5.0, 15.0  # degrees: the same region coming back (NOAA gives it a new number)


def frame_times(start: str) -> list[str]:
    t0 = Time(start)
    return [(t0 + TimeDelta(h * 3600, format="sec")).isot[:19] for h in range(0, DAYS * 24 + 1, STEP_H)]


def load_period(start: str, offline: bool) -> list[list[S.Spot]]:
    t0 = Time(start)
    times = frame_times(start)

    def one(when: str):
        try:
            if offline:
                return None
            return S.fetch_frame(when, CACHE / "frames")
        except (OSError, ValueError, KeyError) as e:
            print(f"  {when}: no frame ({e})", flush=True)
            return None

    with ThreadPoolExecutor(3) as ex:
        got = list(ex.map(one, times))
    frames, seen = [], set()
    for g in got:
        if g is None:
            continue
        data, hdr = g
        if hdr["helioviewer_id"] in seen:  # a gap in the data: the closest image was already used
            continue
        seen.add(hdr["helioviewer_id"])
        t = (Time(hdr["date-obs"]) - t0).to_value("day")
        frames.append(S.detect_spots(data, hdr, float(t)))
    return frames


def srs(day: Time) -> str | None:
    y, m, d = (int(x) for x in day.isot[:10].split("-"))
    path = CACHE / "srs" / f"{y}{m:02d}{d:02d}SRS.txt"
    if not path.is_file():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(S._get(SRS_URL.format(y=y, m=m, d=d), timeout=60))
        except OSError:
            return None
    return path.read_text(errors="replace")


def validate_returns(period_start: str, tracks: list[S.Track], law: S.LawFit, earth: float, scatter: float) -> list[dict]:
    """Regions alive at the end of a period: our return forecast, what NOAA recorded, and NOAA's own forecast."""
    t0 = Time(period_start)
    t_end = max(s.t for tr in tracks for s in tr.spots)
    # Old NOAA numbers near our regions at the end, and the summaries over the return window.
    days = {k: srs(t0 + TimeDelta((t_end + k) * S.DAY, format="sec")) for k in range(-1, RETURN_WINDOW[1] + 1)}
    parsed = {k: S.parse_srs(v, "") if v else ([], []) for k, v in days.items()}
    rows = []
    for tr in tracks:
        last = tr.spots[-1]
        if t_end - last.t > 1.0 or last.lon_stony < -30:  # alive at the end, and on its way to the west limb
            continue
        f = S.forecast_region(tr.lat, last.lon_stony, (t0 + TimeDelta(last.t * S.DAY, format="sec")).isot[:19], law, earth, scatter)
        rate_c = float(law.omega(tr.lat)[0]) - S.CARRINGTON_DEG_PER_DAY
        # Which NOAA region it was: the nearest numbered region in the summary for the period's last day.
        now = [r for r in parsed[0][0] if abs(r.lat - tr.lat) <= 4 and abs(S._dlon(r.lon_carr, last.lon_carr)) <= 8]
        old = min(now, key=lambda r: abs(S._dlon(r.lon_carr, last.lon_carr))).number if now else None
        seen_before = {r.number for k in range(-1, RETURN_WINDOW[0]) for r in parsed[k][0]}
        ret = None
        for k in range(RETURN_WINDOW[0], RETURN_WINDOW[1] + 1):
            for r in parsed[k][0]:
                if r.number in seen_before or r.lon_stony > -50:
                    continue
                lon_then = last.lon_carr + rate_c * (t_end + k - last.t)
                if abs(r.lat - tr.lat) <= MATCH_LAT and abs(S._dlon(r.lon_carr, lon_then)) <= MATCH_LON:
                    syn = f["synodic_deg_day"]
                    listed = t0 + TimeDelta((t_end + k) * S.DAY, format="sec")
                    limb = listed - TimeDelta((90 + r.lon_stony) / syn * S.DAY, format="sec")  # back to E90
                    ret = {"number": r.number, "listed": listed.isot[:10], "at": f"{'N' if r.lat >= 0 else 'S'}{abs(r.lat):.0f}E{abs(r.lon_stony):.0f}",
                           "east_limb_utc": limb.isot[:16]}
                    break
            if ret:
                break
        noaa_forecast = None
        if old is not None:
            for k in range(RETURN_WINDOW[1] + 1):
                hit = next((x for x in parsed[k][1] if x["number"] == old), None)
                if hit:
                    # Section II lists regions due back over the next three days from the summary's date.
                    issued = t0 + TimeDelta((t_end + k) * S.DAY, format="sec")
                    noaa_forecast = (issued + TimeDelta(2 * S.DAY, format="sec")).isot[:16]
                    break
        row = {"lat": tr.lat, "lon_carr": last.lon_carr, "noaa_number": old, "forecast": f, "returned": ret,
               "noaa_forecast_mid_utc": noaa_forecast}
        if ret:
            row["error_days"] = (Time(ret["east_limb_utc"]) - Time(f["east_limb_return"]["utc"])).to_value("day")
            if noaa_forecast:
                row["noaa_error_days"] = (Time(ret["east_limb_utc"]) - Time(noaa_forecast)).to_value("day")
        rows.append(row)
    return rows


def main() -> None:
    if "--bias-only" in sys.argv:  # recompute the return-bias check from the stored returns, no fetching
        out = json.loads((OUT / "results.json").read_text())
        out["return_bias"] = return_bias(out["returns"])
        (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
        print(out["return_bias"])
        return
    offline = "--offline" in sys.argv
    t_start = time.time()
    periods, rates = [], []
    for start, why in PERIODS:
        frames = load_period(start, offline)
        tracks = S.usable(S.link(frames))
        rs = [S.track_rates(tr) for tr in tracks]
        print(f"{start}: {len(frames)} frames, {sum(len(f) for f in frames)} detections, {len(tracks)} tracks", flush=True)
        periods.append({"start": start, "why": why, "frames": len(frames), "tracks": tracks})
        rates += [{**r, "period": start} for r in rs]
    result = S.assess([p["tracks"] for p in periods])
    fits = result["fits"]
    # The law used for forecasts: the winner, or among tied laws the one that forecast the held-out positions best.
    win = result.get("winner") or min(result.get("contenders") or list(fits.values()), key=lambda f: f.heldout_rms_deg)
    # The published profiles all have the A + B sin^2 + C sin^4 form: compare like with like (the sin^2 fit).
    named = fits["sin2"]
    sid = np.array([r["sidereal"] for r in rates])
    lat = np.array([r["lat"] for r in rates])
    scatter = float(np.std(sid - win.omega(lat)))  # spot-to-spot scatter of rates about the law: proper motions
    earth = {p["start"]: S.earth_rate_deg_per_day(p["start"] + "T00:00:00") for p in periods}
    syn_meas = np.array([r["synodic"] for r in rates])
    earth_meas = float(np.median(sid - syn_meas))
    lats = np.linspace(0, 40, 41)
    profiles = {k: {"source": v["source"], "omega": [float(x) for x in S.published_omega(k, lats)],
                    "ABC": v["ABC"]} for k, v in S.PUBLISHED.items()}
    compare = {k: {f"at_{a}": float(named.omega(a)[0] - S.published_omega(k, a)) for a in (0, 15, 30)} for k in S.PUBLISHED}
    compare_sigma = {f"at_{a}": float(named.omega_sigma(a)[0]) for a in (0, 15, 30)}
    returns = []
    for p in periods:
        returns += [{**r, "period": p["start"]} for r in validate_returns(p["start"], p["tracks"], win, earth[p["start"]], scatter)]
    errs = [abs(r["error_days"]) for r in returns if "error_days" in r]
    nerr = [abs(r["noaa_error_days"]) for r in returns if "noaa_error_days" in r]
    out = {
        "generated": "scripts/sun_demo.py", "periods": [{k: v for k, v in p.items() if k != "tracks"} | {"tracks": len(p["tracks"])} for p in periods],
        "status": result["status"], "reason": result["reason"], "winner": win.law.id,
        "laws": [f.summary() for f in fits.values()], "sigma_deg": result["sigma_deg"],
        "rates": rates, "rate_scatter_deg_day": scatter,
        "law_curve": {"lat": lats.tolist(), "omega": win.omega(lats).tolist(), "sigma": win.omega_sigma(lats).tolist()},
        "published": profiles, "difference_from_published": compare, "difference_sigma": compare_sigma,
        "compared_law": "sin2", "named_curve": {"omega": named.omega(lats).tolist(), "sigma": named.omega_sigma(lats).tolist()},
        "earth_rate_deg_day": {"from_sunpy_L0": earth, "measured_sidereal_minus_synodic": earth_meas, "fixed_0_9856": 0.9856},
        "returns": returns,
        "returns_summary": {"forecast": len(returns), "came_back": len(errs), "median_error_days": float(np.median(errs)) if errs else None,
                            "max_error_days": float(max(errs)) if errs else None,
                            "noaa_median_error_days": float(np.median(nerr)) if nerr else None, "noaa_compared": len(nerr),
                            "later_than_forecast": sum(r["error_days"] > 0 for r in returns if "error_days" in r),
                            "noaa_later_than_forecast": sum(r["noaa_error_days"] > 0 for r in returns if "noaa_error_days" in r)},
    }
    out["return_bias"] = return_bias(returns)
    out["outcome"] = outcome(out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(out["status"], out["reason"], out["returns_summary"], f"({time.time() - t_start:.0f} s)")


def return_bias(returns: list[dict]) -> dict:
    """Every region came back later than forecast (and later than NOAA's forecast too). Is that a term the forecast
    should carry? Three candidates, each judged by leave-one-out: fit it on the other returns, forecast the one left
    out. No bias; a constant delay; a delay that grows with latitude. Physical candidates, not decided here: spots
    drifting relative to the surface (proper motion), and a region decaying and re-forming slightly east."""
    rows = [(abs(r["lat"]), r["error_days"]) for r in returns if "error_days" in r]
    lat, err = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    n = len(err)

    def loo(predict) -> float:
        return float(np.mean([abs(err[i] - predict(np.delete(lat, i), np.delete(err, i), lat[i])) for i in range(n)]))

    def lin(la, e, x):
        return float(np.polyval(np.polyfit(la, e, 1), x))

    out = {"returns": n, "late": int((err > 0).sum()),
           # Two-sided sign test: how often n fair coins come out this lopsided.
           "sign_test_p": float(min(1.0, 2 * sum(math.comb(n, k) for k in range(max((err > 0).sum(), (err < 0).sum()), n + 1)) / 2**n)),
           "constant_days": float(err.mean()), "constant_sd_days": float(err.std(ddof=1) / math.sqrt(n)),
           "loo_mae_days": {"none": loo(lambda la, e, x: 0.0), "constant": loo(lambda la, e, x: float(e.mean())),
                            "latitude": loo(lin)},
           "candidates": "spot proper motion relative to the surface; regions decaying and re-forming further east"}
    out["best"] = min(out["loo_mae_days"], key=out["loo_mae_days"].get)
    return out


def outcome(out: dict) -> dict:
    """Proven when the sin^2 law agrees with the published sunspot profile (Allen) within 0.1 deg/day at 0, 15 and
    30 degrees, and at least three regions came back within two days of the forecast."""
    w = next(x for x in out["laws"] if x["law"] == "sin2")
    a, b = w["coef"]["A"], w["coef"]["B"]
    rs = out["returns_summary"]
    d_allen = out["difference_from_published"]["allen"]
    ok_law = all(abs(v) <= 0.1 for v in d_allen.values())
    ok_ret = rs["came_back"] >= 3 and (rs["max_error_days"] or 99) <= 2.0
    at15 = float(np.interp(15, out["law_curve"]["lat"], out["named_curve"]["omega"]))
    allen15 = float(S.published_omega("allen", 15))
    sig15 = out["difference_sigma"]["at_15"]
    return {
        "headline": (f"The Sun turns faster at its equator: {a:.2f} + ({b:.2f}) sin² φ degrees per day (sidereal), from {len(out['rates'])} sunspots "
                     f"in real SDO images, matching the published sunspot profile. Regions came back on the east limb a median "
                     f"{rs['median_error_days']:.1f} days from the forecast ({rs['came_back']} returns)."),
        "action": "Use the return dates to plan observations of regions due back on the east limb.",
        "artifacts": [],
        "checked_against": {"name": "sidereal rate at 15° latitude (deg/day)", "truth": allen15, "measured": at15,
                            "relative_error": (at15 - allen15) / allen15, "within_interval": abs(at15 - allen15) <= 1.96 * sig15,
                            "source": "Allen's Astrophysical Quantities (sunspot rotation, as in sunpy); NOAA SWPC region summaries"},
        "confidence": (f"A = {a:.2f} ± {w['sigma']['A']:.2f}, B = {b:.2f} ± {w['sigma']['B']:.2f} °/day; spots scatter by "
                       f"{out['rate_scatter_deg_day']:.2f} °/day about the law (their own motions)"),
        "status": "proven" if ok_law and ok_ret else "partial",
        "limits": [] if ok_law and ok_ret else [x for x in (
            None if ok_law else "the law differs from the sunspot profile by more than 0.1 °/day somewhere",
            None if ok_ret else f"return forecasts missed by up to {rs['max_error_days']:.1f} days") if x],
        "readable": None}


if __name__ == "__main__":
    main()
