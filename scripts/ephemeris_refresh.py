"""Blind, pre-registered transit predictions for planets whose published ephemeris has gone stale.

  python scripts/ephemeris_refresh.py              # select, measure, refresh, check, register, write data/ephemeris/
  python scripts/ephemeris_refresh.py --no-register

Selection (from the ephemeris-rot list of scripts/timing_demo.py, data/astro/timing/pscomppars_transiting.csv): the
published ephemeris predicts 2027-01-01 to worse than 30 min (1 sigma), V <= 12, period <= 15 days, transit depth
>= 3 ppt (depth from today's NASA Exoplanet Archive row), and TESS SPOC 2-minute light curves taken after the
published reference epoch in at least two observing seasons (one season leaves nothing to hold out).

Per planet: every transit mid-time in those light curves (astro/transits.py, the code used on Kepler), a refreshed
linear ephemeris from them plus the published one as a prior (astro/ephemeris.py), the held-out check (fit on the
earlier seasons, forecast the last one), then the transit windows from 2026-11-15 to 2027-05-15 with 68% and 95%
intervals beside the published ephemeris' own. Only planets that pass the held-out check are registered
(physicslens.prereg); the rest are written down with the reason.

Light curves are cached under var/ephemeris/ (not committed). Free public data, no model calls.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from astropy.time import Time
from physicslens import prereg
from physicslens.astro import ephemeris as E
from physicslens.astro import timing
from physicslens.astro.transits import transit_times

VAR = ROOT / "var" / "ephemeris"
OUT = ROOT / "data" / "ephemeris"
ROT_CSV = ROOT / "data" / "astro" / "timing" / "pscomppars_transiting.csv"
ARCHIVE_CSV = VAR / "pscomppars_now.csv"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLUMNS = ("pl_name,hostname,tic_id,pl_trandep,pl_ratror,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_orbper,"
           "pl_orbpererr1,pl_orbpererr2,pl_trandur,sy_vmag,sy_tmag,pl_tranmid_reflink,pl_orbper_reflink,ra,dec")
ARCHIVE_QUERY = f"select {COLUMNS} from pscomppars where tran_flag=1"
MAST_FILE = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

ROT_DATE = Time("2027-01-01T00:00:00", scale="utc")
WINDOW = (Time("2026-11-15T00:00:00", scale="utc"), Time("2027-05-15T00:00:00", scale="utc"))
MIN_SIGMA_MIN, MAX_VMAG, MAX_PERIOD, MIN_DEPTH_PPT = 30.0, 12.0, 15.0, 3.0
MAX_CONSIDERED = 20  # candidates examined in order of staleness
TARGET = 8
CASE = "transit-timing"
PID_SUFFIX = "2026-11"


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def archive_rows() -> tuple[dict[str, dict[str, str]], str]:
    """Today's archive rows (cached once under var/), and the date they were retrieved."""
    if not ARCHIVE_CSV.exists():
        VAR.mkdir(parents=True, exist_ok=True)
        url = f"{TAP}?query={urllib.parse.quote_plus(ARCHIVE_QUERY)}&format=csv"
        ARCHIVE_CSV.write_bytes(urllib.request.urlopen(url, timeout=180).read())
    text = ARCHIVE_CSV.read_text()
    if not text.startswith("pl_name"):
        raise RuntimeError(f"{ARCHIVE_CSV} is not a CSV from the archive (an error page?): delete it and rerun")
    retrieved = datetime.fromtimestamp(ARCHIVE_CSV.stat().st_mtime, UTC).strftime("%Y-%m-%d")
    return {r["pl_name"]: r for r in csv.DictReader(text.splitlines())}, retrieved


def depth_ppt(row: dict[str, str]) -> float | None:
    if row.get("pl_trandep"):
        return float(row["pl_trandep"]) * 10  # the archive gives percent
    if row.get("pl_ratror"):
        return float(row["pl_ratror"]) ** 2 * 1000
    return None


def published(row: dict[str, str]) -> E.Published:
    def err(a: str, b: str) -> float:  # asymmetric errors: the larger side, as timing.ephemeris_rot does
        return max(abs(float(row[a] or 0)), abs(float(row[b] or 0)))
    return E.Published(float(row["pl_tranmid"]), err("pl_tranmiderr1", "pl_tranmiderr2"), float(row["pl_orbper"]),
                       err("pl_orbpererr1", "pl_orbpererr2"))


def reflink(html: str) -> dict[str, str]:
    """'<a refstr=X href=URL ...>Name</a>' to its name and URL."""
    if "href=" not in html:
        return {"name": html, "url": ""}
    url = html.split("href=")[1].split(" ")[0]
    return {"name": html.split(">")[1].split("<")[0].strip(), "url": url}


def candidates(archive: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    with open(ROT_CSV) as fh:
        ranked = timing.ephemeris_rot(list(csv.DictReader(fh)), ROT_DATE.tdb.jd)
    out = []
    for r in ranked:
        if not (r["sigma_min"] > MIN_SIGMA_MIN and r["vmag"] is not None and r["vmag"] <= MAX_VMAG
                and r["period_days"] <= MAX_PERIOD):
            continue
        row = archive.get(r["planet"])
        d = depth_ppt(row) if row else None
        if d is None or d < MIN_DEPTH_PPT:
            continue
        out.append({**r, "depth_ppt": d, "row": row})
    return out


def sector_of(mission: str) -> int:
    return int(mission.split()[-1])


def download(item: Any, tries: int = 4) -> Any:
    """MAST drops connections now and then; retry with backoff, then let the error through (the caller records it)."""
    for k in range(tries):
        try:
            return item.download(download_dir=str(VAR / "mast"))
        except OSError as err:  # requests' ConnectionError is an OSError
            if k == tries - 1:
                raise
            print(f"  download failed ({err}); retrying", flush=True)
            time.sleep(10 * 2**k)
    return None


def measure(cand: dict[str, Any], pub: E.Published, duration: float) -> dict[str, Any]:
    """Every TESS SPOC 2-min sector after the published reference epoch: its product ids and its mid-times."""
    import lightkurve as lk
    tic = cand["row"]["tic_id"]
    res = lk.search_lightcurve(tic, author="SPOC", exptime=120)
    sectors, rows, skipped = [], [], []
    for i in range(len(res)):
        tab = res.table[i]
        start_bjd = float(tab["t_min"]) + 2400000.5  # MJD (UTC); minutes from BJD_TDB, irrelevant at this scale
        sec = sector_of(str(tab["mission"]))
        if start_bjd <= pub.t0:
            skipped.append({"sector": sec, "reason": "observed before the published reference epoch"})
            continue
        lc = download(res[i])
        lc = lc[lc.quality == 0].remove_nans().normalize()
        t = np.asarray(lc.time.value, float) + 2457000.0  # BTJD to BJD_TDB
        info = {"sector": sec, "tic_id": tic, "obs_id": str(tab["obs_id"]), "obsid": str(tab["obsid"]),
                "product": str(tab["productFilename"]), "url": MAST_FILE + str(tab["dataURI"]),
                "start_bjd": float(t.min()), "end_bjd": float(t.max())}
        try:
            r = transit_times(t, np.asarray(lc.flux.value, float), np.asarray(lc.flux_err.value, float), pub.t0,
                              pub.period, duration, search_half_width=0.1, exp_time=E.TESS_2MIN_EXP_DAYS, refine=True)
        except ValueError as err:
            info["error"] = str(err)
            sectors.append(info)
            continue
        info["n_transits"], info["rejected"] = len(r["rows"]), r["rejected"]
        sectors.append(info)
        rows += [{**row, "sector": sec, "source": f"PhysicsLens fit to TESS SPOC 2-min, {info['obs_id']}"}
                 for row in r["rows"]]
    return {"sectors": sectors, "rows": rows, "skipped": skipped}


def utc(bjd: float) -> str:
    """BJD_TDB to UTC, ignoring the barycentric light time (under 8.3 min): for dates, not for scoring."""
    return Time(bjd, format="jd", scale="tdb").utc.isot[:19] + "Z"


def body_for(cand: dict[str, Any], pub: E.Published, fit: E.Refreshed, held: dict[str, Any], items: list[dict[str, Any]],
             sectors: list[dict[str, Any]], retrieved: str, duration: float) -> dict[str, Any]:
    row, name = cand["row"], cand["planet"]
    first = items[0]
    score_after = utc(first["ci95"][1] + duration / 2)  # the first transit's latest possible end
    t0_ref, p_ref = reflink(row["pl_tranmid_reflink"]), reflink(row["pl_orbper_reflink"])
    inputs = [{"name": f"NASA Exoplanet Archive pscomppars row for {name}", "source": "NASA Exoplanet Archive",
               "url": f"{TAP}?query={urllib.parse.quote_plus(ARCHIVE_QUERY + f' and pl_name={name!r}')}&format=csv",
               "retrieved": retrieved, "values": {k: row[k] for k in ("tic_id", "pl_tranmid", "pl_tranmiderr1",
               "pl_tranmiderr2", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_trandur", "pl_trandep", "sy_vmag")},
               "t0_reference": t0_ref, "period_reference": p_ref}]
    inputs += [{"name": f"TESS sector {s['sector']}, {s['tic_id']}, SPOC 2-min light curve", "source": "MAST (TESS SPOC)",
                "url": s["url"], "retrieved": datetime.now(UTC).strftime("%Y-%m-%d"), "mast_obs_id": s["obs_id"],
                "mast_obsid": s["obsid"], "product": s["product"], "transits_used": s.get("n_transits", 0)}
               for s in sectors]
    eph = {**fit.summary(), "law": "t(N) = t0 + N P, N counted from the published reference transit (N = 0)",
           "published": {"t0_bjd_tdb": pub.t0, "t0_err_days": pub.t0_err, "period_days": pub.period,
                         "period_err_days": pub.period_err},
           "heldout": held, "tic_id": row["tic_id"], "duration_days": duration,
           "method": "weighted least squares on TESS mid-times (errors scaled by their own scatter, never down) plus "
                     "the published t0 and P as independent Gaussian prior rows when they agree within "
                     f"{E.PRIOR_TENSION_MAX:.0f} sigma; intervals from the fit covariance (68%: 1 sigma, 95%: 1.96 sigma)"}
    host = row["hostname"]
    return {"title": f"{name}: transit mid-times from November 2026 to May 2027",
            "question": f"When will {name} transit between 2026-11-15 and 2027-05-15 (BJD_TDB)?",
            "inputs": inputs,
            "prediction": {"kind": "transit_times",
                           "items": [{"label": f"transit N={it['epoch']}", "epoch": it["epoch"], "value": it["value"],
                                      "sigma_days": it["sigma_days"], "ci68": it["ci68"], "ci95": it["ci95"],
                                      "unit": "BJD_TDB", "utc_approx": utc(it["value"]),
                                      "published_ephemeris_value": it["published_ephemeris_value"],
                                      "published_ci95": it["published_ci95"], "shift_min": it["shift_min"]}
                                     for it in items],
                           "ephemeris": eph},
            "scoring": {"rule": ("a hit when an independently measured mid-time (new TESS sector or ExoClock/ETD amateur "
                                 "timing) lies inside the 95% window; reported per transit and as the share of hits. The "
                                 "window is widened in quadrature by the measurement's own 95% error (fixed now, before "
                                 "any measurement: an amateur timing good to 1 to 2 minutes cannot test a window of "
                                 "seconds); how many fall inside the window as registered is reported too. Transits "
                                 "nobody measured count as neither."),
                        "score_after_utc": score_after,
                        "sources_to_check": [
                            (f"TESS SPOC 2-min light curves of {row['tic_id']} observed after registration (MAST, "
                             "lightkurve; scripts/score_predictions.py)"),
                            f"ExoClock observations of {name}: https://www.exoclock.space/database/planets",
                            f"Exoplanet Transit Database (VarAstro) transits of {host}: https://var.astro.cz/ETD"]}}


def process(cand: dict[str, Any], retrieved: str, register: bool) -> dict[str, Any]:
    name, row = cand["planet"], cand["row"]
    pub = published(row)
    rec: dict[str, Any] = {"planet": name, "tic_id": row["tic_id"], "vmag": cand["vmag"], "depth_ppt": cand["depth_ppt"],
                           "published_sigma_2027_min": cand["sigma_min"],
                           "published": {"t0_bjd_tdb": pub.t0, "t0_err_days": pub.t0_err, "period_days": pub.period,
                                         "period_err_days": pub.period_err,
                                         "t0_reference": reflink(row["pl_tranmid_reflink"]),
                                         "period_reference": reflink(row["pl_orbper_reflink"])}}
    if not row.get("pl_trandur"):
        return {**rec, "status": "dropped", "reason": "no transit duration in the archive row: cannot set the search"}
    duration = float(row["pl_trandur"]) / 24
    rec["duration_days"] = duration
    m = measure(cand, pub, duration)
    rec["sectors"], rec["sectors_skipped"] = m["sectors"], m["skipped"]
    rows, rec["transits_rejected"] = E.well_fitted(m["rows"])
    if rows:
        t = np.array([r["bjd_tdb"] for r in rows])
        ep = E.assign_epochs(t, np.array([r["err_days"] for r in rows]), pub)
        for r, n in zip(rows, ep, strict=True):
            r["epoch"] = int(n)
        off = E.season_outliers(ep, t, np.array([r["err_days"] for r in rows]))
        rec["transits_rejected"] += [{"epoch": r["epoch"], "bjd_tdb": r["bjd_tdb"], "sector": r["sector"],
                                      "reason": f"over {E.SEASON_OUTLIER_Z:.0f} robust sigma off the line through the "
                                                "other transits of its season"} for r, o in zip(rows, off, strict=True) if o]
        rows = [r for r, o in zip(rows, off, strict=True) if not o]
    times = np.array([r["bjd_tdb"] for r in rows])
    if len(times) < 4 or E.seasons(times).max() == 0:
        n_seasons = int(E.seasons(times).max()) + 1 if len(times) else 0
        return {**rec, "status": "dropped",
                "reason": f"{len(times)} usable TESS transits in {n_seasons} season(s) after the published epoch: "
                          "nothing to hold out"}
    errs = np.array([r["err_days"] for r in rows])
    epochs = np.array([r["epoch"] for r in rows])
    rec["midtimes"] = [{k: r[k] for k in ("epoch", "bjd_tdb", "err_days", "sector", "source", "chi2_red")}
                       for r in rows]
    fit = E.refresh(epochs, times, errs, pub)
    held = E.heldout(epochs, times, errs, pub)
    rec["refreshed"], rec["heldout"] = fit.summary(), held
    # The shift at 2027-01-01: the transit nearest that date, refreshed against published.
    n27 = round((ROT_DATE.tdb.jd - fit.t0) / fit.period)
    t_r, s_r = fit.predict(np.array([n27]))
    t_p, s_p = pub.predict(np.array([n27]))
    rec["at_2027_01_01"] = {"epoch": int(n27), "refreshed_bjd": float(t_r[0]), "refreshed_sigma_min": float(s_r[0] * 1440),
                            "published_bjd": float(t_p[0]), "published_sigma_min": float(s_p[0] * 1440),
                            "shift_min": float((t_r[0] - t_p[0]) * 1440),
                            "shift_in_published_sigma": float((t_r[0] - t_p[0]) / s_p[0])}
    items = E.windows(fit, pub, WINDOW[0].tdb.jd, WINDOW[1].tdb.jd, duration)
    rec["windows"] = items
    if not held["passed"]:
        return {**rec, "status": "not registered", "reason": f"held-out check failed: {held['reason']}"}
    if fit.scale > E.MAX_ERROR_SCALE:
        return {**rec, "status": "not registered",
                "reason": (f"the TESS mid-times scatter {fit.scale:.1f} times their own errors (limit "
                           f"{E.MAX_ERROR_SCALE:.0f}): the timing is not understood, though the held-out check passed "
                           f"with errors inflated that much ({held['reason']})")}
    pid = f"{slug(name)}-{PID_SUFFIX}"
    path = prereg.path_for(CASE, pid)
    if path.exists():
        # Registered by an earlier run: never again. Recomputing from the same cached data must give the same windows.
        doc = json.loads(path.read_text())
        same = [it["value"] for it in doc["prediction"]["items"]] == [w["value"] for w in items]
        sha = next((r["sha256"] for r in prereg.index_rows() if r["case"] == CASE and r["id"] == pid), None)
        return {**rec, "status": "registered", "prediction": str(path.relative_to(ROOT)), "sha256": sha,
                "registered_utc": doc["created_utc"], "recomputed_matches_registered": same,
                "note": "registered by an earlier run; not re-registered (a prediction is never edited)"}
    if not register:
        return {**rec, "status": "passed, not registered (--no-register)"}
    body = body_for(cand, pub, fit, held, items, [s for s in m["sectors"] if s.get("n_transits")], retrieved, duration)
    path = prereg.register(CASE, pid, body)
    return {**rec, "status": "registered", "prediction": str(path.relative_to(ROOT)), "sha256": prereg.sha256(path)}


def write_outputs(records: list[dict[str, Any]], retrieved: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in records if r.get("refreshed")]
    eph_rows = []
    for r in ok:
        s = r["refreshed"]
        eph_rows.append({"name": r["planet"], "t0_bjd_tdb": s["pivot_bjd_tdb"], "t0_unc": s["pivot_err_days"],
                         "period_days": s["period_days"], "period_unc": s["period_err_days"],
                         "duration_hours": r["duration_days"] * 24,
                         "depth_mmag": -2.5 * np.log10(1 - r["depth_ppt"] / 1000) * 1000,
                         "n_tess_transits": s["n_new_times"],
                         "first_tess_bjd": min(m["bjd_tdb"] for m in r["midtimes"]),
                         "last_tess_bjd": max(m["bjd_tdb"] for m in r["midtimes"]),
                         "published_t0_bjd_tdb": r["published"]["t0_bjd_tdb"],
                         "published_period_days": r["published"]["period_days"],
                         # The label says whether the planet was registered, not only whether the held-out check
                         # passed: a check can pass on inflated errors (KELT-8 b) and still fail the scatter gate.
                         "source": f"PhysicsLens refresh from TESS SPOC 2-min; held-out "
                                   f"{'passed' if r['heldout']['passed'] else 'FAILED'}; "
                                   f"{'registered' if r.get('status') == 'registered' else 'not registered: ' + str(r.get('reason') or r.get('status', ''))}"})
    (OUT / "refreshed_ephemerides.csv").write_text(E.exoclock_csv(eph_rows))
    (OUT / "tess_midtimes.csv").write_text("".join(
        E.midtimes_csv(r["planet"], r["midtimes"]).split("\n", 1)[1] if i else E.midtimes_csv(r["planet"], r["midtimes"])
        for i, r in enumerate(ok)))
    win = [("planet,epoch,mid_bjd_tdb,sigma_minutes,ci68_lo,ci68_hi,ci95_lo,ci95_hi,mid_utc_approx,"
            "published_mid_bjd_tdb,published_ci95_lo,published_ci95_hi,shift_minutes")]
    for r in ok:
        for w in r["windows"]:
            win.append(",".join([r["planet"], str(w["epoch"]), f"{w['value']:.6f}", f"{w['sigma_days'] * 1440:.2f}",
                                 *(f"{v:.6f}" for v in (*w["ci68"], *w["ci95"])), utc(w["value"]),
                                 f"{w['published_ephemeris_value']:.6f}",
                                 *(f"{v:.6f}" for v in w["published_ci95"]), f"{w['shift_min']:.1f}"]))
    (OUT / "windows_2026-11_2027-05.csv").write_text("\n".join(win) + "\n")
    summary = {"generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "archive_retrieved": retrieved,
               "selection": {"published_sigma_at_2027_01_01_min_over": MIN_SIGMA_MIN, "vmag_at_most": MAX_VMAG,
                             "period_days_at_most": MAX_PERIOD, "depth_ppt_at_least": MIN_DEPTH_PPT,
                             "tess": "SPOC 2-min sectors observed after the published reference epoch, two seasons or more",
                             "considered": len(records), "target": TARGET},
               "rules": {"heldout_chi2_per_transit_max": timing.HELDOUT_CHI2, "prior_tension_max_sigma": E.PRIOR_TENSION_MAX,
                         "season_gap_days": E.SEASON_GAP_DAYS},
               "planets": records}
    (OUT / "results.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")


def main() -> None:
    warnings.filterwarnings("ignore")
    archive, retrieved = archive_rows()
    cands = candidates(archive)
    print(f"{len(cands)} candidates pass the catalogue cuts; examining up to {MAX_CONSIDERED}")
    records, registered = [], 0
    for c in cands[:MAX_CONSIDERED]:
        if registered >= TARGET:
            break
        print(f"{c['planet']} (sigma {c['sigma_min']:.0f} min at 2027-01-01, depth {c['depth_ppt']:.1f} ppt)", flush=True)
        try:
            rec = process(c, retrieved, register="--no-register" not in sys.argv)
        except OSError as err:  # MAST unreachable after retries: this planet is skipped, and the record says so
            rec = {"planet": c["planet"], "status": "dropped", "reason": f"TESS download failed after retries: {err}"}
        registered += rec["status"] == "registered"
        h = rec.get("heldout", {})
        print(f"  {rec['status']}: {rec.get('reason') or h.get('reason', '')}"
              f"{' shift ' + format(rec['at_2027_01_01']['shift_min'], '.1f') + ' min' if 'at_2027_01_01' in rec else ''}",
              flush=True)
        records.append(rec)
    write_outputs(records, retrieved)
    print(f"{registered} registered; wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
