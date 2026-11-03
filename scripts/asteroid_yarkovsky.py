"""The Yarkovsky push on Apophis from 18 years of optical observations, held out and compared with JPL.

  python scripts/asteroid_yarkovsky.py       # run on ADA: each fit integrates 18 years many times (8 workers)

Sunlight absorbed and re-emitted by a spinning asteroid pushes it along its orbit: the transverse acceleration
A2 (1 AU / r)^2 of orbit computers (JPL's convention, A2 in AU/day^2). One night cannot see it; years can. Here:
- Minor Planet Center optical observations of 99942 Apophis (ADES, each with its stated uncertainty), thinned to at
  most two per observatory per night so one busy night does not outweigh a year. Radar is not used (JPL's solution
  also has 20 delays and 30 Dopplers), so the optical arc alone sets the precision here.
- The orbit: Sun, planets and the Moon (as for every asteroid here), the Sun's relativistic term, and A2 when fitted.
  No asteroid perturbers (JPL includes the 16 largest), stated as a limit.
- Held out: fitted on 2004 to 2016, with and without A2, each forecasting the 2017 to 2022 observations it never saw.
  A2 counts as found only if it forecasts them better and its fitted value is several sigma from zero.
- Then the whole arc with A2 free, against JPL SBDB's published A2 for Apophis (fetched, with its sigma).
Free: MPC and JPL public APIs, cached under var/astro/ast/. Writes data/astro/asteroids/yarkovsky.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from astropy.time import Time
from physicslens.astro import asteroids as A

CACHE = ROOT / "var" / "astro" / "ast"
OUT = ROOT / "data" / "astro" / "asteroids" / "yarkovsky.json"
DESIG = "99942"
SPLIT = "2017-01-01"  # fitted before, forecast after
PER_NIGHT = 2
A2_UNIT = 1e-14  # the fitted parameter is A2 / 1e-14 AU/day^2, so all seven are of order one after scaling
WORKERS = int(os.environ.get("PL_WORKERS", "8"))


def cached_json(name: str, fetch):
    p = CACHE / name
    if not p.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fetch()))
    return json.loads(p.read_text())


def mpc_ades() -> list[dict]:
    def get():
        req = urllib.request.Request("https://data.minorplanetcenter.net/api/get-obs", method="GET",
                                     data=json.dumps({"desigs": [DESIG], "output_format": ["ADES_DF"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())[0]["ADES_DF"]
    return cached_json("apophis_ades.json", get)


def jpl_a2() -> dict:
    def get():
        with urllib.request.urlopen(f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={DESIG}", timeout=120) as r:
            d = json.loads(r.read())
        o = d["orbit"]
        a2 = next(p for p in o["model_pars"] if p["name"] == "A2")
        return {"a2": float(a2["value"]), "sigma": float(a2["sigma"]), "orbit_id": o.get("orbit_id"),
                "soln_date": o.get("soln_date"), "n_obs_used": o.get("n_obs_used"),
                "n_del_obs_used": o.get("n_del_obs_used"), "n_dop_obs_used": o.get("n_dop_obs_used")}
    return cached_json("apophis_sbdb.json", get)


def start_state(epoch_tdb: float) -> list[float]:
    def get():
        from astroquery.jplhorizons import Horizons
        v = Horizons(id=DESIG, id_type="smallbody", location="@sun", epochs=epoch_tdb).vectors(refplane="earth")
        return [float(v[k][0]) for k in ("x", "y", "z", "vx", "vy", "vz")]
    return cached_json(f"apophis_start_{epoch_tdb:.5f}.json", get)


def thin(rows: list[dict]) -> list[dict]:
    """At most PER_NIGHT observations per observatory per night, the best-measured first."""
    keep, seen = [], {}
    for r in sorted(rows, key=lambda r: float(r.get("rmsra") or 9) + float(r.get("rmsdec") or 9)):
        k = (r["stn"], r["obstime"][:10])
        if seen.get(k, 0) < PER_NIGHT:
            seen[k] = seen.get(k, 0) + 1
            keep.append(r)
    return sorted(keep, key=lambda r: r["obstime"])


# Worker state, set once per process (fork): the observations, the planets and the epoch.
_W: dict = {}


def _init(obs, bodies, epoch, fit_a2):
    _W.update(obs=obs, bodies=bodies, epoch=epoch, fit_a2=fit_a2)


def _resid(p: np.ndarray) -> np.ndarray:
    obs = _W["obs"]
    a2 = p[6] * A2_UNIT if _W["fit_a2"] else 0.0
    ra_c, dec_c = A.predict(p[:6], _W["epoch"], obs, _W["bodies"], a2=a2, relativity=True)
    a, b = A.residuals(ra_c, dec_c, obs)
    return np.concatenate([a / obs.sigma_ra, b / obs.sigma_dec])


STEPS = np.array([1e-8] * 3 + [1e-10] * 3 + [0.05])


def fit(obs: A.Observations, bodies, epoch: float, p0: np.ndarray, fit_a2: bool, rounds: int = 3) -> dict:
    """Weighted least squares (Gauss-Newton with a damped step) on the angles, outliers beyond 3 sigma set aside."""
    n_par = 7 if fit_a2 else 6
    p = np.array(p0[:n_par], float)
    used = np.ones(len(obs.ra), bool)
    for _ in range(rounds):
        sub = obs.subset(used)
        with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(sub, bodies, epoch, fit_a2)) as pool:
            for _ in range(8):
                pts = [p] + [p + np.eye(n_par)[i] * STEPS[i] for i in range(n_par)]
                rs = list(pool.map(_resid, pts))
                r0 = rs[0]
                J = np.column_stack([(rs[i + 1] - r0) / STEPS[i] for i in range(n_par)])
                dp = np.linalg.lstsq(J, -r0, rcond=None)[0]
                lam, best = 1.0, None
                for _ in range(6):  # step halving: accept the first step that lowers chi-square
                    q = p + lam * dp
                    r1 = list(pool.map(_resid, [q]))[0]
                    if r1 @ r1 < r0 @ r0:
                        best = q
                        break
                    lam /= 2
                if best is None:
                    break
                done = np.all(np.abs(best - p) < STEPS[:n_par] * 1e-3)
                p = best
                if done:
                    break
            r = list(pool.map(_resid, [p]))[0]
        _init(obs, bodies, epoch, fit_a2)
        full = _resid(p)
        k = len(obs.ra)
        norm = np.hypot(full[:k], full[k:]) / math.sqrt(2)
        new = norm <= A.OUTLIER_SIGMA
        if np.array_equal(new, used):
            break
        used = new
    dof = max(len(r) - n_par, 1)
    chi2 = float(r @ r) / dof
    cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2)
    return {"p": p, "cov": cov, "used": used, "chi2_per_dof": chi2, "n_used": int(used.sum()), "chi2_total": float(r @ r)}


def forecast(sol: dict, obs: A.Observations, bodies, epoch: float, fit_a2: bool) -> dict:
    _init(obs, bodies, epoch, fit_a2)
    r = _resid(sol["p"])
    k = len(obs.ra)
    norm = np.hypot(r[:k], r[k:]) / math.sqrt(2)
    keep = norm <= 5 * A.OUTLIER_SIGMA  # gross blunders in the later data say nothing about the model
    a, b = r[:k][keep] * obs.sigma_ra[keep], r[k:][keep] * obs.sigma_dec[keep]
    return {"n": int(keep.sum()), "rms_arcsec": float(np.sqrt(np.mean(a ** 2 + b ** 2) / 2)),
            "chi2_per_obs": float(np.mean(norm[keep] ** 2)), "median_miss_arcsec": float(np.median(np.hypot(a, b)))}


def main() -> None:
    t_start = time.time()
    codes = A.parse_obscodes((CACHE / "ObsCodes.html").read_text(errors="replace"))
    rows = thin([r for r in mpc_ades() if r.get("Obstype") in (None, "optical")])
    obs = A.from_ades_rows(rows, codes)  # older observations state no uncertainty: 1 arcsec until weighted below
    t = obs.t_utc_jd
    early = t < Time(SPLIT).jd
    print(f"{len(t)} observations ({early.sum()} before {SPLIT}, {(~early).sum()} after)", flush=True)
    epoch = float(np.round(A.tdb(np.array([t[early].mean()]))[0], 1))
    bodies = A.planets(t.min() - 2, t.max() + 2)
    s0 = np.array(start_state(epoch) + [0.0])
    jpl = jpl_a2()

    # Observatories that state no uncertainty are weighted by their own scatter about a gravity-only fit to the early
    # years (no A2, no later data), as orbit computers weight by station; stated uncertainties are kept.
    stated = np.array([bool(r.get("rmsra")) for r in sorted(
        [r for r in rows if r.get("stn") in codes and r.get("ra") and r.get("dec")], key=lambda r: r["obstime"])])
    if len(stated) != len(t):
        stated = np.zeros(len(t), bool)
    base = fit(obs.subset(early), bodies, epoch, s0, False, rounds=2)
    _init(obs, bodies, epoch, False)
    r_all = _resid(base["p"])
    k = len(t)
    off = np.hypot(r_all[:k] * obs.sigma_ra, r_all[k:] * obs.sigma_dec) / math.sqrt(2)  # arcsec
    weights = {}
    for stn in sorted(set(obs.stn)):
        m = (obs.stn == stn) & early & ~stated
        if m.sum() >= 5:
            core = off[m][off[m] < 5.0]  # blunders say nothing about a station's precision
            if len(core) >= 5:
                weights[str(stn)] = float(max(np.sqrt(np.mean(core ** 2)), 0.2))
    for i in np.where(~stated)[0]:
        w = weights.get(str(obs.stn[i]))
        if w:
            obs.sigma_ra[i] = obs.sigma_dec[i] = w
    print(f"{len(weights)} observatories weighted by their scatter; {stated.sum()} observations state their own", flush=True)

    held = {}
    for name, fa2 in (("without_a2", False), ("with_a2", True)):
        sol = fit(obs.subset(early), bodies, epoch, s0, fa2)
        fc = forecast(sol, obs.subset(~early), bodies, epoch, fa2)
        held[name] = {"chi2_per_dof": sol["chi2_per_dof"], "n_used": sol["n_used"], "forecast": fc}
        if fa2:
            held[name]["a2"] = float(sol["p"][6] * A2_UNIT)
            held[name]["a2_sigma"] = float(math.sqrt(sol["cov"][6, 6]) * A2_UNIT)
        print(name, json.dumps(held[name])[:400], f"({time.time() - t_start:.0f} s)", flush=True)

    full0 = fit(obs, bodies, epoch, s0, False)
    full = fit(obs, bodies, epoch, s0, True)
    a2, a2s = float(full["p"][6] * A2_UNIT), float(math.sqrt(full["cov"][6, 6]) * A2_UNIT)
    n = 2 * full["n_used"]
    bic = {"without_a2": full0["chi2_total"] + 6 * math.log(n), "with_a2": full["chi2_total"] + 7 * math.log(n)}
    better = held["with_a2"]["forecast"]["rms_arcsec"] < held["without_a2"]["forecast"]["rms_arcsec"]
    found = better and abs(a2) > 3 * a2s
    out = {
        "generated": "scripts/asteroid_yarkovsky.py", "object": "99942 Apophis",
        "observations": {"n": len(t), "first": Time(t.min(), format="jd").isot[:10], "last": Time(t.max(), format="jd").isot[:10],
                         "before_split": int(early.sum()), "after_split": int((~early).sum()), "split": SPLIT,
                         "thinning": f"at most {PER_NIGHT} per observatory per night", "source": "MPC get-obs API, ADES",
                         "with_stated_uncertainty": int(stated.sum()), "observatories_weighted": len(weights)},
        "model": "Sun, planets and the Moon (DE-based, astropy), the Sun's relativistic term, light-time; A2 (1 AU/r)^2 transverse",
        "held_out": held,
        "full_arc": {"a2": a2, "a2_sigma": a2s, "chi2_per_dof": full["chi2_per_dof"], "n_used": full["n_used"],
                     "chi2_per_dof_without": full0["chi2_per_dof"], "delta_bic": bic["without_a2"] - bic["with_a2"]},
        "jpl": jpl,
        "agreement_sigma": (a2 - jpl["a2"]) / math.hypot(a2s, jpl["sigma"]),
        "found": bool(found),
        "limits": ["optical astrometry only (JPL also uses radar delays and Dopplers)",
                   "no asteroid perturbers (JPL includes the 16 most massive)"],
        "seconds": round(time.time() - t_start),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("full_arc", "jpl", "agreement_sigma", "found")}, indent=1))


if __name__ == "__main__":
    main()
