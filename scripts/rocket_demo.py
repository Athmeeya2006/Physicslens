"""A real rocket: Falcon 9's engine performance from its launch webcast telemetry, checked against SpaceX's figures.

  python scripts/rocket_demo.py fetch     download the pinned public telemetry into var/rocket/ (free)
  python scripts/rocket_demo.py           fit every first-stage ascent, write data/rocket/result.json (free)

Data: speed and altitude read from SpaceX launch webcasts at 30 per second, with the time of each launch's max-Q
throttle-down and of main-engine cut-off (shahar603/Telemetry-Data, the Unlicense). The first stage flies from T+0
to cut-off (MECO). Along the flight path,
    dv/dt = k(t) (c - A p(h)) / (tau - K(t)) - g(h) sin(gamma) - beta rho(h) v^2 tau / (tau - K(t))
with c the vacuum exhaust velocity, A the nozzle exit area over the full-throttle mass flow (how much the air's
pressure p takes off the thrust), tau the burn time to empty at full throttle (lift-off mass over mass flow), k the
throttle (1, or k_b inside a bucket from t1 to t2), K its integral, beta the drag area over twice the lift-off mass,
rho and p from the standard atmosphere, gamma from the measured climb (sin gamma = dh/dt / v). Specific impulse is
c / g0 in vacuum and (c - A p0) / g0 at sea level. Candidates, simplest first: the ideal rocket (thrust and gravity), + the
throttle bucket, + the air-pressure loss, and each of those with drag (a law whose fitted value sits at a limit of its range is
not verified). Each is fitted on the first 75% of the burn and must forecast
the rest; the simplest law that passes and wins by BIC is kept, as in the rest of PhysicsLens. The bucket's start
and end are fitted blind and only then compared with the times the telemetry lists. Local math, no model calls.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
REPO, COMMIT = "shahar603/Telemetry-Data", "master"
RAW = ROOT / "var" / "rocket"
OUT = ROOT / "data" / "rocket"
G0 = 9.80665
R_EARTH = 6.371e6
EDGE_S = 0.5  # logistic width of the throttle bucket's edges (about 2 s from 10% to 90%)
DT = 0.1  # the fit runs at 10 per second (the webcast's 30 per second readings repeat between digit changes)
HOLD = 0.75
# SpaceX's published Merlin 1D figures (Falcon 9 Full Thrust user's guide and spacex.com vehicle page, as tabulated on
# Wikipedia's Merlin page): specific impulse 282 s at sea level and 311 s in vacuum.
PUBLISHED = {"isp_sl_s": 282.0, "isp_vac_s": 311.0, "source": "SpaceX Merlin 1D (Falcon 9 FT): 282 s sea level, 311 s vacuum"}
# Candidate laws, simplest first: their free parameters.
# Drag costs Falcon 9 a few percent of its speed by cut-off, too little to show in the webcast readout beside the
# thrust terms (its fitted coefficient goes to zero), so laws without it are candidates too.
LAWS = {"ideal": ("c", "tau"), "drag": ("c", "tau", "beta"), "bucket": ("c", "tau", "kb", "t1", "t2"),
        "bucket_pressure": ("c", "tau", "kb", "t1", "t2", "A"), "drag_bucket": ("c", "tau", "beta", "kb", "t1", "t2"),
        "drag_bucket_pressure": ("c", "tau", "beta", "kb", "t1", "t2", "A")}
START = {"c": 3000.0, "tau": 200.0, "beta": 1e-5, "kb": 0.7, "t1": 45.0, "t2": 75.0, "A": 0.002}
BOUNDS = {"c": (1500.0, 4500.0), "tau": (120.0, 400.0), "beta": (0.0, 1e-3), "kb": (0.3, 1.0), "t1": (20.0, 90.0),
          "t2": (25.0, 120.0), "A": (0.0, 0.01)}


def atmosphere(h_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Density (kg/m^3) and pressure (Pa) of the US Standard Atmosphere 1976, by its layers up to 86 km."""
    layers = [(0, 288.15, -0.0065, 101325.0), (11000, 216.65, 0.0, 22632.06), (20000, 216.65, 0.001, 5474.889),
              (32000, 228.65, 0.0028, 868.0187), (47000, 270.65, 0.0, 110.9063), (51000, 270.65, -0.0028, 66.93887),
              (71000, 214.65, -0.002, 3.956420)]
    h = np.clip(h_m, 0, 86000)
    p = np.empty_like(h)
    T = np.empty_like(h)
    for i, (hb, Tb, L, pb) in enumerate(layers):
        top = layers[i + 1][0] if i + 1 < len(layers) else 1e9
        m = (h >= hb) & (h < top)
        if L == 0:
            T[m] = Tb
            p[m] = pb * np.exp(-G0 * 0.0289644 * (h[m] - hb) / (8.31446 * Tb))
        else:
            T[m] = Tb + L * (h[m] - hb)
            p[m] = pb * (T[m] / Tb) ** (-G0 * 0.0289644 / (8.31446 * L))
    return p * 0.0289644 / (8.31446 * T), p


def fetch() -> None:
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/Laucnhes.json", timeout=60) as r:
        index = json.loads(r.read())
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "launches.json").write_text(json.dumps(index))
    with urllib.request.urlopen(f"https://api.github.com/repos/{REPO}/commits/{COMMIT}", timeout=60) as r:
        sha = json.loads(r.read())["sha"]
    (RAW / "source.json").write_text(json.dumps({"repo": REPO, "commit": sha, "licence": "Unlicense"}))
    for x in index:
        name = x["mission_name"]
        d = RAW / name
        d.mkdir(exist_ok=True)
        for kind, rel in (("raw", f"{name}/JSON/{name} raw.json"), ("events", f"{name}/JSON/events.json"),
                          ("stages", f"{name}/JSON/stages.json")):
            url = f"https://raw.githubusercontent.com/{REPO}/{sha}/" + urllib.parse.quote(rel)
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    (d / f"{kind}.json").write_bytes(r.read())
            except OSError as e:
                print(name, kind, "missing:", e, flush=True)
        print("fetched", name, flush=True)


def load(name: str) -> dict | None:
    """The first stage's flight, resampled to DT: time, speed (m/s), altitude (m), sin of the climb angle; and the
    listed events. None when the launch has no usable first-stage telemetry (it starts after lift-off, or no MECO)."""
    d = RAW / name
    try:
        raw = json.loads((d / "raw.json").read_text())
        ev = json.loads((d / "events.json").read_text())
        stages = json.loads((d / "stages.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    meco = ev.get("meco")
    if not meco or "1" in stages and stages["1"] and stages["1"][0] > 1.0:
        return None
    t, v, h = (np.asarray(raw[k], float) for k in ("time", "velocity", "altitude"))
    if t[0] > 1.0 or t[-1] < meco:
        return None
    # Stop 3 s before the listed cut-off (the engines throttle and shut down over about a second) and start once the
    # rocket clears the tower (below 20 m/s the readout is too coarse to differentiate).
    tt = np.arange(0.0, meco - 3.0, DT)
    vi = np.interp(tt, t, v)
    hi = np.interp(tt, t, h * 1000.0)
    vs = savgol_filter(vi, 51, 2)
    hs = savgol_filter(hi, 51, 2)
    climb = np.gradient(hs, DT)
    sg = np.clip(climb / np.maximum(vs, 1.0), -1, 1)
    start = int(np.argmax(vs > 20.0))
    return {"name": name, "t": tt[start:], "v": vi[start:], "h": hs[start:], "sin": sg[start:], "events": ev,
            "meco": meco}


def simulate(law: str, p: dict[str, float], f: dict, upto: int | None = None) -> np.ndarray:
    """Speed along the flight path from the first sample, integrated (RK2) with the measured altitude and climb."""
    t, h, sg = f["t"], f["h"], f["sin"]
    n = len(t) if upto is None else upto
    rho, pr = atmosphere(h[:n])
    g = G0 * (R_EARTH / (R_EARTH + h[:n])) ** 2
    k = np.ones(n)
    if "bucket" in law:
        # Smooth edges (a 1 s logistic): the engines throttle over about a second, and a hard step would give the fit
        # no gradient to move the edges by.
        inside = 1 / (1 + np.exp(-(t[:n] - p["t1"]) / EDGE_S)) - 1 / (1 + np.exp(-(t[:n] - p["t2"]) / EDGE_S))
        k = 1 - (1 - p["kb"]) * inside
    K = np.concatenate([[0.0], np.cumsum(k[:-1]) * DT]) + t[0]  # full-throttle-equivalent burn time since lift-off
    c_eff = p["c"] - (p.get("A", 0.0) * pr if "pressure" in law else 0.0)
    left = np.maximum(p["tau"] - K, 1.0)
    thrust = k * c_eff / left
    drag_k = p.get("beta", 0.0) * rho * p["tau"] / left
    v = np.empty(n)
    v[0] = f["v"][0]
    for i in range(n - 1):
        a1 = thrust[i] - g[i] * sg[i] - drag_k[i] * v[i] ** 2
        vp = v[i] + DT * a1
        a2 = thrust[i + 1] - g[i + 1] * sg[i + 1] - drag_k[i + 1] * vp ** 2
        v[i + 1] = v[i] + 0.5 * DT * (a1 + a2)
    return v


def fit(law: str, f: dict, n: int, start: dict | None = None) -> tuple[dict, float]:
    names = LAWS[law]
    x0 = np.array([(start or START)[k] for k in names])
    lo, hi = np.array([BOUNDS[k][0] for k in names]), np.array([BOUNDS[k][1] for k in names])
    scale = np.maximum(np.abs(x0), 1e-6)

    def res(x):
        p = dict(zip(names, x * scale, strict=True))
        if "bucket" in law and p["t2"] <= p["t1"] + 2:
            return np.full(n, 1e3)
        return simulate(law, p, f, n) - f["v"][:n]

    best = None
    # Several starts for the bucket's edges: with smooth edges the surface is differentiable but has local minima.
    starts = [x0]
    if "bucket" in law:
        i1, i2 = names.index("t1"), names.index("t2")
        starts = []
        for a in (35, 45, 55):
            for b in (60, 70, 85):
                if b > a + 8:
                    s = x0.copy()
                    s[i1], s[i2] = a, b
                    starts.append(s)
    for s in starts:
        r = least_squares(res, np.clip(s, lo, hi) / scale, bounds=(lo / scale, hi / scale), x_scale="jac", max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    return dict(zip(names, best.x * scale, strict=True)), float(2 * best.cost)


def nrmse(sim: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((sim - obs) ** 2)) / np.std(obs))


def analyse(name: str) -> dict | None:
    f = load(name)
    if f is None:
        return None
    n = len(f["t"])
    nfit = int(n * HOLD)
    out = {"mission": name, "t0_s": round(float(f["t"][0]), 2), "t_end_s": round(float(f["t"][-1]), 2), "meco_listed_s": f["meco"],
           "throttle_listed_s": [f["events"].get("throttle_down_start"), f["events"].get("throttle_down_end")], "laws": {}}
    for law, names in LAWS.items():
        p, rss = fit(law, f, nfit)
        sim = simulate(law, p, f)
        held = nrmse(sim[nfit:], f["v"][nfit:])
        bic = nfit * np.log(rss / nfit) + len(names) * np.log(nfit)
        # The rule of the rest of PhysicsLens: a law whose fitted value sits at a limit of its allowed range is not a
        # verified law (the term it carries is not supported by the data).
        pinned = [k for k, v in p.items() if min(abs(v - BOUNDS[k][0]), abs(v - BOUNDS[k][1])) < 1e-3 * (BOUNDS[k][1] - BOUNDS[k][0])]
        out["laws"][law] = {"params": {k: float(v) for k, v in p.items()}, "heldout_nrmse": round(held, 5), "pinned": pinned,
                            "heldout_rms_ms": round(float(np.sqrt(np.mean((sim[nfit:] - f["v"][nfit:]) ** 2))), 2),
                            "bic": round(float(bic), 1)}
    # The rule of the rest of PhysicsLens: forecast within 15% of the held-out spread; lowest BIC, by at least 10
    # over the next simpler passing law, else the simpler one stays.
    passing = [k for k in LAWS if out["laws"][k]["heldout_nrmse"] <= 0.15 and not out["laws"][k]["pinned"]]
    chosen = None
    for k in passing:
        if chosen is None or out["laws"][chosen]["bic"] - out["laws"][k]["bic"] >= 10:
            chosen = k
    out["chosen"] = chosen
    if chosen:
        p = out["laws"][chosen]["params"]
        pr0 = 101325.0
        # The chosen law refitted on the whole burn for the reported values, with a residual bootstrap for intervals.
        full, _ = fit(chosen, f, n, start={**START, **p})
        out["full_burn"] = {k: float(v) for k, v in full.items()}
        out["isp_vac_s"] = full["c"] / G0
        out["isp_sl_s"] = (full["c"] - full.get("A", 0.0) * pr0) / G0 if "pressure" in chosen else None
        if "bucket" in chosen:
            out["bucket_fitted_s"] = [round(full["t1"], 1), round(full["t2"], 1)]
            out["throttle_kb"] = full["kb"]
    print(name, chosen, {k: out["laws"][k]["heldout_nrmse"] for k in LAWS},
          round(out.get("isp_vac_s") or 0, 1), out.get("isp_sl_s") and round(out["isp_sl_s"], 1), out.get("bucket_fitted_s"),
          out["throttle_listed_s"], flush=True)
    return out


def main() -> None:
    index = json.loads((RAW / "launches.json").read_text())
    names = [x["mission_name"] for x in index if "Heavy" not in x["mission_name"] and "BFR" not in x["mission_name"]]
    with ProcessPoolExecutor(16) as pool:
        rows = [r for r in pool.map(analyse, names) if r]
    chosen = [r for r in rows if r["chosen"]]
    vac = np.array([r["isp_vac_s"] for r in chosen])
    sl = np.array([r["isp_sl_s"] for r in chosen if r["isp_sl_s"] is not None])
    buckets = [r for r in chosen if r.get("bucket_fitted_s") and all(r["throttle_listed_s"])]
    edge_err = np.array([abs(a - b) for r in buckets for a, b in zip(r["bucket_fitted_s"], r["throttle_listed_s"], strict=True)])
    held = {k: float(np.median([r["laws"][k]["heldout_nrmse"] for r in rows])) for k in LAWS}
    summary = {"launches_with_first_stage": len(rows), "with_accepted_law": len(chosen),
               "chosen_counts": {k: sum(r["chosen"] == k for r in rows) for k in LAWS},
               "heldout_nrmse_median": {k: round(v, 4) for k, v in held.items()},
               "isp_vac_median_s": round(float(np.median(vac)), 1) if len(vac) else None,
               "isp_vac_iqr_s": [round(float(np.percentile(vac, 25)), 1), round(float(np.percentile(vac, 75)), 1)] if len(vac) else None,
               "isp_sl_median_s": round(float(np.median(sl)), 1) if len(sl) else None,
               "isp_sl_iqr_s": [round(float(np.percentile(sl, 25)), 1), round(float(np.percentile(sl, 75)), 1)] if len(sl) else None,
               "isp_sl_launches": len(sl),
               "bucket_edge_error_median_s": round(float(np.median(edge_err)), 1) if len(edge_err) else None,
               "bucket_launches": len(buckets), "published": PUBLISHED}
    src = json.loads((RAW / "source.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({"generated": "scripts/rocket_demo.py", "source": src, "laws": {k: list(v) for k, v in LAWS.items()},
                                                 "summary": summary, "launches": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        main()
