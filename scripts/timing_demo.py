"""Transit timing on real data: WASP-12b's decaying orbit, Kepler-88's hidden planet, and the ephemeris-rot list.

  python scripts/timing_demo.py [--skip-nbody]
  python scripts/timing_demo.py --family 40   # the Kepler-88 companion refitted from 40 spread starts

Reads data/astro/timing/ (published WASP-12b timings, Kepler-88 b transit times measured here from Kepler light
curves, the NASA Exoplanet Archive table) and writes data/astro/timing/results.json plus transit windows (CSV, ICS).
Free and local: no model calls.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from astropy.time import Time
from physicslens.astro import timing
from physicslens.astro.ttv import Setup, companion_candidates, fit_companion, fit_from, nbody_times, spread_starts

DATA = ROOT / "data" / "astro" / "timing"
NOW = Time("2026-11-01T00:00:00", scale="utc")
ROT_DATE = Time("2027-01-01T00:00:00", scale="utc")
# Published values compared against (each checked in the paper's own source, see data/astro/timing/SOURCES.md).
YEE_2020_RATE = (-29.0, 2.0)  # ms per year
NESVORNY_2013 = {"Pc_days": 22.3383, "Pc_sigma": 0.0036, "mc_earth": 215.9, "mc_sigma": 7.6, "p_ttv_days": 630.0}
KOI142_MSTAR = 1.022


def read_rows(name: str) -> list[dict]:
    with open(DATA / name) as fh:
        return list(csv.DictReader(fh))


def law_rows(r: dict) -> list[dict]:
    return [{"law": f.law.id, "title": f.law.title, "latex": f.law.latex, "heldout_chi2": f.heldout_chi2,
             "heldout_rms_min": f.heldout_rms_min, "bic": f.bic, "params": f.params, "sigma": f.sigma()} for f in r["fits"]]


def oc_points(d: timing.Timings, base) -> list[list[float]]:
    lin = timing.LAWS["linear"].model(base, d.epoch, d.occultation)
    return [[int(e), float(t), float((t - m) * 1440), float(s * 1440), bool(o)]
            for e, t, m, s, o in zip(d.epoch, d.time, lin, d.err, d.occultation, strict=True)]


def wasp12() -> dict:
    d = timing.Timings.from_rows(read_rows("wasp12b_yee2020.csv"))
    r = timing.assess(d)
    win = r["winner"]
    rate, sig = timing.period_change_ms_per_year(next(f for f in r["fits"] if f.law.id == "decay"))
    rows = timing.next_windows(win, NOW.tdb.jd, 20)
    (DATA / "wasp12b_windows.csv").write_text(timing.windows_csv("WASP-12 b", rows))
    (DATA / "wasp12b_windows.ics").write_text(timing.windows_ics("WASP-12 b", rows))
    result = {"system": "WASP-12 b", "status": r["status"], "reason": r["reason"], "winner": win.law.id if win else None,
              "laws": law_rows(r), "train_until_bjd": r["train_until_bjd"], "n_train": r["n_train"], "n_test": r["n_test"],
              "period_change_ms_per_year": [rate, sig], "published": {"source": "Yee et al. 2020, ApJL 888, L5",
                                                                      "rate_ms_per_year": list(YEE_2020_RATE)},
              "oc": oc_points(d, timing.linear_start(d)), "next_windows": rows[:10]}
    err = rate - YEE_2020_RATE[0]
    result["outcome"] = {
        "headline": (f"WASP-12b's orbit is shrinking: each orbit is {abs(rate):.0f} ± {sig:.0f} ms shorter per year, "
                     "and the decaying-orbit law forecast the later transits that no other law could."),
        "action": f"Schedule observations from the refreshed ephemeris: next transit {rows[0]['utc_approx']} UTC "
                  f"± {rows[0]['sigma_min']:.1f} min. Download the windows (CSV or calendar).",
        "artifacts": [{"kind": "windows-csv"}, {"kind": "windows-ics"}],
        "checked_against": {"name": "dP/dt", "truth": YEE_2020_RATE[0], "measured": rate, "relative_error": err / abs(YEE_2020_RATE[0]),
                            "within_interval": abs(err) <= 1.96 * np.hypot(sig, YEE_2020_RATE[1]),
                            "source": "Yee et al. 2020 (ApJL 888, L5)"},
        "confidence": f"dP/dt between {rate - 1.96 * sig:.1f} and {rate + 1.96 * sig:.1f} ms/yr (95%)",
        "status": "proven" if abs(err) <= 1.96 * np.hypot(sig, YEE_2020_RATE[1]) else "partial", "readable": None}
    return result


def kepler88(nbody: bool) -> dict:
    d = timing.Timings.from_rows(read_rows("kepler88b_transits.csv"))
    r = timing.assess(d, laws=("linear", "decay", "sinusoid"))
    sin = next(f for f in r["fits"] if f.law.id == "sinusoid")
    p_b = sin.params["P"]
    p_ttv = abs(sin.params["Nttv"]) * p_b
    result = {"system": "Kepler-88 b (KOI-142.01)", "status": r["status"], "reason": r["reason"], "laws": law_rows(r),
              "measured_ttv_period_days": p_ttv, "measured_ttv_amplitude_h": abs(sin.params["A"]) * 24,
              "transits_used": len(d.time), "oc": oc_points(d, timing.linear_start(d)),
              "published": {"source": "Nesvorny et al. 2013, ApJ 777, 3", **NESVORNY_2013}}
    cands = [c for c in companion_candidates(p_b, p_ttv, js=(2, 3))]
    result["candidates"] = cands
    if not nbody:
        return result
    order = np.argsort(d.time)
    cut = d.time[order[int(len(order) * timing.TRAIN_FRACTION)]]
    tr, te = d.time < cut, d.time >= cut
    s = Setup(KOI142_MSTAR, int(d.epoch.min()))
    fits = []
    for c in cands:
        t0 = time.time()
        f = fit_companion(d.epoch[tr], d.time[tr], d.err[tr], c, s, p_b)
        if f is None:
            continue
        pred = nbody_times(f.theta, d.epoch[te], s)
        scale = max(1.0, np.sqrt(f.chi2 / max(tr.sum() - len(f.theta), 1)))
        f.heldout_chi2 = float(np.nanmean(((pred - d.time[te]) / (d.err[te] * scale)) ** 2))
        f.heldout_rms_min = float(np.sqrt(np.nanmean(((pred - d.time[te]) * 1440) ** 2)))
        fits.append(f)
        print(f"  {c['resonance']} {c['where']} P={c['period']:.3f}: train chi2 {f.chi2:.0f}, held-out rms "
              f"{f.heldout_rms_min:.1f} min, Pc {f.theta[4]:.4f}, mc {f.theta[8]:.0f} Me ({time.time() - t0:.0f} s)", flush=True)
    result["companion_fits"] = [f.summary() for f in fits]
    sinus_rms = next(x["heldout_rms_min"] for x in result["laws"] if x["law"] == "sinusoid")
    good = sorted([f for f in fits if f.heldout_rms_min < sinus_rms], key=lambda f: f.bic)
    if good:
        best = good[0]
        result["companion"] = best.summary()
        result["companion_degenerate_with"] = [f.summary() for f in good[1:] if f.bic - best.bic < timing.DECISIVE_DELTA_BIC]
        result["outcome"] = kepler88_outcome(result)
    return result


# Starting points refined further: the coarse fits stop after 300 evaluations, which is enough to rank them but not to
# converge the best few.
FAMILY_REFINE = 10


def _family_member(args: tuple) -> dict | None:
    p0, cand, p_b, nfev = args
    d = timing.Timings.from_rows(read_rows("kepler88b_transits.csv"))
    tr, te = split(d)
    s = Setup(KOI142_MSTAR, int(d.epoch.min()))
    f = fit_from(p0, d.epoch[tr], d.time[tr], d.err[tr], cand, s, p_b, max_nfev=nfev)
    if f is None:
        return None
    pred = nbody_times(f.theta, d.epoch[te], s)
    scale = max(1.0, np.sqrt(f.chi2 / max(tr.sum() - len(f.theta), 1)))
    f.heldout_chi2 = float(np.nanmean(((pred - d.time[te]) / (d.err[te] * scale)) ** 2))
    f.heldout_rms_min = float(np.sqrt(np.nanmean(((pred - d.time[te]) * 1440) ** 2)))
    return f.summary()


def split(d: timing.Timings) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(d.time)
    cut = d.time[order[int(len(order) * timing.TRAIN_FRACTION)]]
    return d.time < cut, d.time >= cut


def kepler88_family(result: dict, n: int, workers: int = 12) -> dict:
    """Refit the winning companion from n starting points spread over the period window and both eccentricities,
    refine the best of them, and keep every fit that matches the training transits as well as the best (chi-square
    within the decisive BIC margin; every fit has the same 9 parameters) and passes the held-out rule that admitted
    the best (it forecasts the later transits better than the best simple law). The range they span is the
    uncertainty the timing data actually allow, which the one fit's formal errors understated."""
    from multiprocessing import Pool

    best = result["companion"]
    cand, p_b = best["candidate"], best["Pb"]
    d = timing.Timings.from_rows(read_rows("kepler88b_transits.csv"))
    starts = spread_starts(cand, p_b, float(d.time[np.argmin(d.epoch)]), n)
    t0 = time.time()
    names = ["T0", "Pb", "kb", "hb", "Pc", "lam_c", "kc", "hc", "mc_earth"]
    with Pool(workers) as pool:
        coarse = [f for f in pool.map(_family_member, [(p0, cand, p_b, 300) for p0 in starts]) if f]
        coarse.sort(key=lambda f: f["chi2"])
        fine = [f for f in pool.map(_family_member, [(np.array([f[k] for k in names]), cand, p_b, 1500)
                                                       for f in coarse[:FAMILY_REFINE]]) if f]
    fits = fine + coarse[FAMILY_REFINE:] + [dict(best)]
    sinus = next(x["heldout_rms_min"] for x in result["laws"] if x["law"] == "sinusoid")
    floor = min(f["chi2"] for f in fits)
    kept = [f for f in fits if f["chi2"] - floor < timing.DECISIVE_DELTA_BIC and f["heldout_rms_min"] < sinus]
    top = min(f["heldout_rms_min"] for f in kept)
    pc = [f["Pc"] for f in kept]
    mc = [f["mc_earth"] for f in kept]
    pub = NESVORNY_2013
    print(f"  family: {len(fits)} fits from {n} starts, {len(kept)} kept, Pc {min(pc):.3f} to {max(pc):.3f}, "
          f"mc {min(mc):.0f} to {max(mc):.0f} ({time.time() - t0:.0f} s)", flush=True)
    return {"starts": n, "fits": len(fits), "refined": FAMILY_REFINE, "kept": len(kept),
            "rule": (f"training chi-square within {timing.DECISIVE_DELTA_BIC:g} of the lowest (the decisive BIC margin) and "
                     "a held-out forecast better than the best simple law's"),
            "best_heldout_rms_min": top, "Pc_range": [min(pc), max(pc)], "mc_earth_range": [min(mc), max(mc)],
            "eb_range": [min(f["eb"] for f in kept), max(f["eb"] for f in kept)],
            "ec_range": [min(f["ec"] for f in kept), max(f["ec"] for f in kept)],
            "published_period_inside": bool(min(pc) <= pub["Pc_days"] <= max(pc)),
            "published_mass_inside": bool(min(mc) - pub["mc_sigma"] <= pub["mc_earth"] <= max(mc) + pub["mc_sigma"]),
            "members": [{k: f[k] for k in ("Pc", "mc_earth", "eb", "ec", "chi2", "heldout_rms_min")} for f in kept],
            "all_fits": sorted(({k: f[k] for k in ("Pc", "mc_earth", "eb", "ec", "chi2", "heldout_rms_min")} for f in fits),
                               key=lambda f: f["chi2"])}


def kepler88_outcome(result: dict) -> dict:
    """Proven only when the N-body law passes the same held-out rule as every law and the period agrees within the
    combined interval; otherwise partial, with the misses stated."""
    bs = result["companion"]
    pc, mc = bs["Pc"], bs["mc_earth"]
    pub = NESVORNY_2013
    err_p = (pc - pub["Pc_days"]) / pub["Pc_days"]
    within = abs(pc - pub["Pc_days"]) <= 1.96 * float(np.hypot(bs["Pc_sigma"], pub["Pc_sigma"]))
    passes = bs["heldout_chi2"] <= timing.HELDOUT_CHI2
    sinus = next(x["heldout_rms_min"] for x in result["laws"] if x["law"] == "sinusoid")
    off_sigmas = abs(pc - pub["Pc_days"]) / max(bs["Pc_sigma"], 1e-12)
    fam = result.get("companion_family")
    if fam:
        # The range over every fit that forecasts as well as the best is the uncertainty; the one fit's formal errors
        # are not (they ignore the mass-eccentricity trade-off).
        (p_lo, p_hi), (m_lo, m_hi) = fam["Pc_range"], fam["mc_earth_range"]
        within = fam["published_period_inside"]
        found = (f"{p_lo:.2f} to {p_hi:.2f} days and {m_lo:.0f} to {m_hi:.0f} Earth masses" if fam["kept"] > 1 else
                 f"about {p_lo:.2f} days and {m_lo:.0f} Earth masses")
        conf = ((f"the range over {fam['kept']} of {fam['fits']} N-body fits from {fam['starts']} spread starting points "
                 f"that fit the earlier transits as well as the best and pass the same held-out rule; the published period "
                 f"is {'inside' if within else 'outside'} it. ") if fam["kept"] > 1 else
                (f"{fam['fits']} N-body fits from {fam['starts']} starting points spread over the period window and both "
                 f"eccentricities find one basin: no other fit comes within the decisive margin of it, so the timing data "
                 f"allow no wider family here, and the published period ({abs(p_lo / pub['Pc_days'] - 1):.1%} away) stays "
                 f"outside. "))
    elif within:
        found = f"about {pc:.2f} days and {mc:.0f} ± {bs['mc_earth_sigma']:.0f} Earth masses"
        conf = (f"period {pc:.3f} ± {bs['Pc_sigma']:.3f} d, mass {mc - 1.96 * bs['mc_earth_sigma']:.0f} to "
                f"{mc + 1.96 * bs['mc_earth_sigma']:.0f} Earth masses (95%, at the fitted eccentricities). ")
    else:
        # The fit's own errors do not cover the published answer, so they are not the uncertainty: timing data trade
        # the companion's mass against its eccentricity, and a family of solutions fits. Say so, rather than print a ±
        # that misses by many of itself.
        found = f"roughly {pc:.1f} days and {mc:.0f} Earth masses"
        conf = (f"one solution of a family: the fit's formal errors (± {bs['Pc_sigma']:.3f} d, ± {bs['mc_earth_sigma']:.0f} Earth "
                f"masses) are too small, since the published period is {off_sigmas:.0f} of them away ({abs(err_p):.1%}); in timing "
                "data the companion's mass trades off against its eccentricity. ")
    return {
        "headline": (f"An unseen planet pulls Kepler-88 b: {found}, found from transit times alone. It forecasts the later "
                     f"transits to {(fam['best_heldout_rms_min'] if fam else bs['heldout_rms_min']):.0f} min; the best "
                     f"simple law misses by {sinus:.0f} min."),
        "action": "Confirm with radial velocities, or schedule transit observations from the N-body forecast.",
        "artifacts": [],
        "checked_against": {"name": "companion period", "truth": pub["Pc_days"], "measured": pc, "relative_error": err_p,
                            "within_interval": bool(within), "source": "Nesvorny et al. 2013 (later confirmed by radial velocities)"},
        "confidence": conf + f"Published: {pub['Pc_days']} d, {pub['mc_earth']} ± {pub['mc_sigma']} Earth masses.",
        "status": "proven" if within and passes else "partial",
        "limits": [] if passes else [(f"held-out chi-square {bs['heldout_chi2']:.1f} per transit is above the {timing.HELDOUT_CHI2:.0f} "
                                      "every law must meet: the timing errors (about 3 min) are smaller than what this fit explains")],
        "readable": None}


def rot() -> dict:
    rows = read_rows("pscomppars_transiting.csv")
    ranked = timing.ephemeris_rot(rows, ROT_DATE.tdb.jd)
    return {"date": ROT_DATE.isot[:10], "source": "NASA Exoplanet Archive, pscomppars (fetched 2026-11-01)",
            "planets": len(rows), "with_uncertainties": len(ranked),
            "over_10_min": sum(r["over_10_min"] for r in ranked), "over_1_h": sum(r["over_1_h"] for r in ranked),
            "bright_over_1_h": sum(1 for r in ranked if r["over_1_h"] and r["vmag"] is not None and r["vmag"] < 12),
            "top": [r for r in ranked if r["vmag"] is not None and r["vmag"] < 13][:200]}


def add_curves(out: dict) -> None:
    """Each law's O-C curve on a fine grid of epochs, for the charts (cheap: no refitting)."""
    for key, path in (("wasp12b", "wasp12b_yee2020.csv"), ("kepler88b", "kepler88b_transits.csv")):
        res = out.get(key)
        if not res:
            continue
        d = timing.Timings.from_rows(read_rows(path))
        base = timing.linear_start(d)
        lo, hi = int(d.epoch.min()), int(d.epoch.max())
        grid = np.linspace(lo, hi + (hi - lo) * 0.15, 300)
        lin = timing.LAWS["linear"].model(base, grid, np.zeros(len(grid), bool))
        for row in res["laws"]:
            law = timing.LAWS[row["law"]]
            theta = np.array([row["params"][k] for k in law.params])
            row["curve"] = [[float(e), float((m - li) * 1440)] for e, m, li in
                            zip(grid, law.model(theta, grid, np.zeros(len(grid), bool)), lin, strict=True)]
        comp = res.get("companion")
        if comp:
            s = Setup(KOI142_MSTAR, int(d.epoch.min()))
            theta = np.array([comp[k] for k in ("T0", "Pb", "kb", "hb", "Pc", "lam_c", "kc", "hc", "mc_earth")])
            ep = np.arange(lo, int(hi + (hi - lo) * 0.15))
            tt = nbody_times(theta, ep, s)
            li = timing.LAWS["linear"].model(base, ep, np.zeros(len(ep), bool))
            comp["curve"] = [[int(e), float((t - m) * 1440)] for e, t, m in zip(ep, tt, li, strict=True) if np.isfinite(t)]


def main() -> None:
    if "--family" in sys.argv:
        out = json.loads((DATA / "results.json").read_text())
        out["kepler88b"]["companion_family"] = kepler88_family(out["kepler88b"], int(sys.argv[sys.argv.index("--family") + 1]))
        out["kepler88b"]["outcome"] = kepler88_outcome(out["kepler88b"])
        (DATA / "results.json").write_text(json.dumps(out, indent=1, default=float))
        print(out["kepler88b"]["outcome"]["headline"])
        return
    if "--curves-only" in sys.argv:
        out = json.loads((DATA / "results.json").read_text())
        if out["kepler88b"].get("companion"):
            out["kepler88b"]["outcome"] = kepler88_outcome(out["kepler88b"])
        add_curves(out)
        (DATA / "results.json").write_text(json.dumps(out, indent=1, default=float))
        return
    nbody = "--skip-nbody" not in sys.argv
    out = {"generated": "scripts/timing_demo.py", "wasp12b": wasp12(), "rot": rot()}
    print("WASP-12b:", out["wasp12b"]["status"], out["wasp12b"]["period_change_ms_per_year"])
    out["kepler88b"] = kepler88(nbody)
    add_curves(out)
    (DATA / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("Kepler-88:", out["kepler88b"].get("companion", "no companion fit"))


if __name__ == "__main__":
    main()
