"""Classical baselines on the same tracks as the gallery results: what you get without the agent or the verification.

For each gallery clip it rebuilds the exact coordinates PhysicsLens fitted (from the stored track, the fit's origin,
scale and geometry), checks them by re-simulating PhysicsLens's own law on them, then runs two baselines on the first
75% and forecasts the last 25%, the same split PhysicsLens is judged on:

  * curve_fit of the textbook law: a simple harmonic swing, a damped harmonic oscillator, a drag-free parabola, or a
    Kepler ellipse (elements and GM), chosen by the clip's family;
  * PySINDy alone: library defaults with polynomials up to degree 3, on the state (q, dq/dt) by finite differences.

  python scripts/baselines.py            # writes benchmark/results/baselines.json and benchmark/results/BASELINES.md

Local math on committed files: free, and deterministic.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.physics.coords import to_coordinates
from physicslens.physics.custom import spec_for
from physicslens.physics.fit import TRAIN_FRACTION, simulate
from physicslens.sigproc.trajectory import Trajectory

OUT = ROOT / "benchmark" / "results"
GM_SUN_AU3_D2 = 2.9591220828559115e-4  # JPL DE440, AU^3/day^2
GM_JUPITER_KM3_S2 = 1.26686531e8  # JPL Horizons, Jupiter system
R_JUPITER_KM = 71492.0  # the equatorial radius the Jupiter demo used as its unit
TRUTHS: dict[str, dict[str, Any]] = {
    "gallery-pendulum": {"name": "g", "value": 9.81, "unit": "m/s^2",
                         "source": "standard gravity at the filming site (Aachen, 9.811); length 0.5 m given"},
    "gallery-apophis": {"name": "mu", "value": GM_SUN_AU3_D2, "unit": "AU^3/day^2", "source": "the Sun's GM, JPL DE440"},
    "gallery-halley": {"name": "mu", "value": GM_SUN_AU3_D2, "unit": "AU^3/day^2", "source": "the Sun's GM, JPL DE440"},
    "gallery-jupiter": {"name": "mu", "value": GM_JUPITER_KM3_S2 * 3600**2 / R_JUPITER_KM**3, "unit": "R_J^3/hour^2",
                        "source": "Jupiter's GM, JPL Horizons, in Jupiter radii and hours"},
    "gallery-spring": None,  # no ruler and no independent stiffness: forecasts only
    "gallery-paperbag": None,  # the scale comes from g itself: drag has no independent truth
}
FAMILIES = ("pendulum", "oscillator", "projectile", "plane")  # each has a textbook law below
# The parameter each row compares when there is no truth to name it.
COMPARED = {"gallery-spring": "omega", "gallery-paperbag": "k"}


def coordinates(result: dict[str, Any]):
    fs, clip = result["fit_state"], result["clip"]
    trk = np.asarray(result["real_track_px"], dtype=float)
    trk = trk[trk[:, 0] >= fs["first_frame"]]
    rate = clip["fps"] * clip["slow_motion"]
    ox, oy = fs["origin_px"]
    xy = np.column_stack([(trk[:, 1] - ox) / fs["px_per_m"], (oy - trk[:, 2]) / fs["px_per_m"]])
    traj = Trajectory((trk[:, 0] - fs["first_frame"]) / rate, xy, trk[:, 0].astype(int), rate)
    return to_coordinates(traj, result["family"], fs.get("ctx") or None)


def nrmse(pred: np.ndarray, q: np.ndarray, n_train: int) -> float:
    """PhysicsLens's held-out score: RMSE over the last quarter, of a forecast fitted on the first three quarters."""
    if pred is None or not np.all(np.isfinite(pred)):
        return math.inf
    return float(np.sqrt(np.mean((pred[n_train:] - q[n_train:]) ** 2)) / (np.std(q) + 1e-12))


# ---------- textbook laws ----------

def _harmonic(t, A, w, phi, c):
    return A * np.cos(w * t + phi) + c


def _damped(t, A, beta, w, phi, c):
    return A * np.exp(-beta * t) * np.cos(w * t + phi) + c


def _first_frequency(t: np.ndarray, x: np.ndarray) -> float:
    """A starting angular frequency, so curve_fit starts on the right swing: the spectrum's peak is too coarse on a
    few seconds of data, so frequencies around it are scanned with a linear sinusoid fit and the best one is kept."""
    dt = float(np.median(np.diff(t)))
    f = np.fft.rfftfreq(len(x), dt)[1:]
    p = np.abs(np.fft.rfft(x - x.mean()))[1:]
    peak = 2 * math.pi * float(f[np.argmax(p)])

    def rss(w: float) -> float:
        A = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
        return float(np.sum((A @ np.linalg.lstsq(A, x, rcond=None)[0] - x) ** 2))
    return min(np.linspace(0.3 * peak, 3.0 * peak, 400), key=rss)


def textbook_oscillation(c, n_train: int, damped: bool) -> tuple[np.ndarray, float, float]:
    t, x = c.t, c.q[:, 0]
    w0 = _first_frequency(t[:n_train], x[:n_train])
    amp = float(np.ptp(x[:n_train]) / 2)
    if damped:
        p, cov = curve_fit(_damped, t[:n_train], x[:n_train], p0=[amp, 0.05, w0, 0.0, x.mean()], maxfev=20000)
        w, sw = math.hypot(p[2], p[1]), math.sqrt(cov[2, 2] + cov[1, 1])  # the undamped frequency
        return _damped(t, *p)[:, None], w, sw
    p, cov = curve_fit(_harmonic, t[:n_train], x[:n_train], p0=[amp, w0, 0.0, x.mean()], maxfev=20000)
    return _harmonic(t, *p)[:, None], abs(p[1]), math.sqrt(cov[1, 1])


def textbook_parabola(c, n_train: int) -> tuple[np.ndarray, float, float]:
    t, q = c.t, c.q
    px = np.polyfit(t[:n_train], q[:n_train, 0], 1)
    py, cov = np.polyfit(t[:n_train], q[:n_train, 1], 2, cov=True)
    pred = np.column_stack([np.polyval(px, t), np.polyval(py, t)])
    return pred, -2 * py[0], 2 * math.sqrt(cov[0, 0])


def _kepler_xy(t, gm, a, e, varpi, m0, sense):
    n = math.sqrt(gm / a**3) * sense
    m = m0 + n * t
    E = m.copy()
    for _ in range(60):  # Newton on Kepler's equation; converges for e < 0.99 from E = M
        E = E - (E - e * np.sin(E) - m) / (1 - e * np.cos(E))
    xp, yp = a * (np.cos(E) - e), a * math.sqrt(1 - e * e) * np.sin(E)
    cw, sw = math.cos(varpi), math.sin(varpi)
    return np.column_stack([cw * xp - sw * yp, sw * xp + cw * yp])


def textbook_kepler(c, n_train: int) -> tuple[np.ndarray, float, float]:
    """A Kepler ellipse about the origin (the Sun or Jupiter): conic and mean motion for a start, then curve_fit."""
    t, q = c.t[:n_train], c.q[:n_train]
    r, th = np.hypot(q[:, 0], q[:, 1]), np.arctan2(q[:, 1], q[:, 0])
    A = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    k, *_ = np.linalg.lstsq(A, 1 / r, rcond=None)
    p_semi, e = 1 / k[0], math.hypot(k[1], k[2]) / k[0]
    varpi = math.atan2(k[2], k[1])
    e = min(e, 0.98)
    a = p_semi / (1 - e * e)
    sense = 1.0 if np.mean(q[:, 0] * np.gradient(q[:, 1], t) - q[:, 1] * np.gradient(q[:, 0], t)) > 0 else -1.0
    nu = sense * (th - varpi)
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    M = np.unwrap(E - e * np.sin(E))
    n_fit, m0 = np.polyfit(t, M, 1)
    gm0 = (n_fit**2) * a**3

    def model(tt, gm, aa, ee, vv, mm):
        return _kepler_xy(tt, gm, aa, ee, vv, mm, sense).ravel()

    p, cov = curve_fit(model, t, q.ravel(), p0=[gm0, a, e, varpi, m0 if sense > 0 else -m0],
                       bounds=([gm0 * 0.2, a * 0.5, 0.0, -10, -100], [gm0 * 5, a * 2, 0.995, 10, 100]), maxfev=20000)
    return _kepler_xy(c.t, *p, sense), float(p[0]), float(math.sqrt(cov[0, 0]))


# ---------- PySINDy alone ----------

class _TooLong(Exception):
    pass


def _integrate(model, x0: np.ndarray, t: np.ndarray, span: float, seconds: float = 20.0) -> np.ndarray | None:
    """The learned equations from the first state, stopped (None) if they blow up or take too long: both are results."""
    import time

    from scipy.integrate import solve_ivp
    d = len(x0) // 2
    deadline = time.monotonic() + seconds

    def rhs(_t, y):
        if time.monotonic() > deadline:
            raise _TooLong
        return model.predict(y[None, :])[0]

    def blown(_t, y):
        return 1e3 * span - float(np.max(np.abs(y)))
    blown.terminal = True
    try:
        sol = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, method="LSODA", rtol=1e-6, atol=1e-9, events=blown)
    except _TooLong:
        return None
    return sol.y.T[:, :d] if sol.success and sol.y.shape[1] == len(t) else None


def sindy_alone(c, n_train: int) -> tuple[np.ndarray | None, dict[str, float], str]:
    import pysindy as ps
    t, q = c.t, c.q
    dt = float(np.median(np.diff(t)))
    v = np.gradient(q, dt, axis=0)  # finite differences, as PySINDy itself would take them
    x = np.hstack([q, v])
    d = q.shape[1]
    names = [f"q{i}" for i in range(d)] + [f"v{i}" for i in range(d)]
    model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x[:n_train], t=t[:n_train], feature_names=names)
        coefs = model.coefficients()
        pred = _integrate(model, x[0], t, span=float(np.ptp(x, axis=0).max()))
    terms = dict(zip(model.get_feature_names(), range(coefs.shape[1]), strict=True))
    accel = {name: float(coefs[d + i, j]) for i in range(d) for name, j in terms.items() if abs(coefs[d + i, j]) > 0
             for name in [f"{names[d + i]}': {name}"]}
    eqs = "; ".join(model.equations(precision=4)[d:])
    return pred, accel, eqs


def physicslens_row(result: dict[str, Any]) -> dict[str, Any]:
    fs = result["fit_state"]
    win = next(r for r in result["ranking"] if r["model_id"] == fs["model_id"])
    return {"status": result["status"], "law": (result.get("equation") or {}).get("title"),
            # A tie has no single law's parameters, but the values every tied law agrees on are measured either way.
            "parameters": [{k: p.get(k) for k in ("name", "value", "std", "unit")}
                           for p in result.get("parameters") or result.get("agreed_parameters") or []],
            "heldout_nrmse": win.get("heldout_nrmse")}


def run_clip(gid: str) -> dict[str, Any]:
    result = json.loads((ROOT / "gallery" / gid / "result.json").read_text())
    c = coordinates(result)
    n_train = int(len(c.t) * TRAIN_FRACTION)
    fs = result["fit_state"]
    check = simulate(spec_for(fs), fs["params"], np.asarray(fs["y0"], dtype=float), c.t, fs.get("ctx") or {})
    reproduce = float(np.sqrt(np.mean((check - c.q) ** 2)) / np.std(c.q))
    row: dict[str, Any] = {"clip": gid, "family": result["family"], "samples": len(c.t), "truth": TRUTHS.get(gid),
                           "coordinates_check_nrmse": reproduce, "physicslens": physicslens_row(result)}
    family, L = result["family"], (fs.get("ctx") or {}).get("length_m")
    if family == "pendulum":
        pred, w, sw = textbook_oscillation(c, n_train, damped=False)
        tb = {"law": "simple harmonic swing, theta = A cos(w t + phi)", "name": "g", "value": w * w * L,
              "std": 2 * w * L * sw, "unit": "m/s^2"}
    elif family == "oscillator":
        pred, w, sw = textbook_oscillation(c, n_train, damped=True)
        tb = {"law": "damped harmonic oscillator", "name": "omega", "value": w, "std": sw, "unit": "rad/s"}
    elif family == "projectile":
        pred, g, sg = textbook_parabola(c, n_train)
        tb = {"law": "drag-free parabola", "name": "g", "value": g, "std": sg, "unit": "m/s^2",
              "note": "the textbook parabola has no drag term, so it has no k; its own g (with the scale set by "
                      f"g = 9.81) comes out at {g:.4g}"}
    elif family == "plane":
        pred, gm, sgm = textbook_kepler(c, n_train)
        tb = {"law": "Kepler ellipse (elements and GM)", "name": "mu", "value": gm, "std": sgm,
              "unit": (result.get("units") or {}).get("length", "") + "^3/" + (result.get("units") or {}).get("time", "") + "^2"}
    tb["heldout_nrmse"] = nrmse(pred, c.q, n_train)
    row["textbook_curve_fit"] = tb

    pred, accel, eqs = sindy_alone(c, n_train)
    sindy: dict[str, Any] = {"law": eqs, "heldout_nrmse": nrmse(pred, c.q, n_train)}
    if family in ("pendulum", "oscillator"):
        k = accel.get("v0': q0")
        if k is not None and k < 0:
            sindy.update({"name": "g" if family == "pendulum" else "omega",
                          "value": -k * L if family == "pendulum" else math.sqrt(-k)})
    elif family == "projectile":
        k = accel.get("v0': v0")  # linear drag: x'' = -k x'
        if k is not None:
            sindy.update({"name": "k", "value": -k})
    else:
        sindy["note"] = "a polynomial library cannot express an inverse-square pull, so no GM can be read off"
    row["pysindy_alone"] = sindy
    return row


def closer(row: dict[str, Any]) -> str:
    truth = row["truth"]
    if not truth:
        return "no independent truth"
    pl = next((p["value"] for p in row["physicslens"]["parameters"] if p["name"] == truth["name"]), None)
    cands = {"PhysicsLens": pl, "textbook curve_fit": row["textbook_curve_fit"].get("value"),
             "PySINDy alone": row["pysindy_alone"].get("value")}
    errs = {k: abs(v - truth["value"]) / truth["value"] for k, v in cands.items() if v is not None}
    return min(errs, key=errs.get) if errs else "none"


def fmt(v: float | None, s: float | None = None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.5g} ± {s:.2g}" if s else f"{v:.5g}"


def _cell(name: str, v: float | None, s: float | None, e: float) -> str:
    return f"{name} = {fmt(v, s)}; " + (f"held-out {e:.3g}" if math.isfinite(e) else "diverged")


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Classical baselines on the gallery tracks", "",
             ("Generated by `scripts/baselines.py` from the committed gallery results: the same coordinates and the same "
              "75/25 split PhysicsLens is judged on. Held-out error is the RMSE of the forecast over the last quarter "
              "divided by the coordinates' standard deviation (lower is better; diverged: the learned equations blew "
              "up)."), "",
             "| Clip | Truth | PhysicsLens (verified) | Textbook law, curve_fit | PySINDy alone | Closest to truth |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        t = r["truth"]
        pl = r["physicslens"]
        name = t["name"] if t else COMPARED.get(r["clip"], pl["parameters"][0]["name"] if pl["parameters"] else "")
        plp = next((p for p in pl["parameters"] if p["name"] == name), None)
        tb, sy = r["textbook_curve_fit"], r["pysindy_alone"]
        cell = partial(_cell, name)
        lines.append(f"| {r['clip'].removeprefix('gallery-')} | "
                     f"{fmt(t['value']) + ' ' + t['unit'] if t else 'none'} | "
                     f"{cell(plp and plp['value'], plp and plp['std'], pl['heldout_nrmse'])} | "
                     f"{cell(tb.get('value') if tb.get('name') == name else None, tb.get('std') if tb.get('name') == name else None, tb['heldout_nrmse'])} | "
                     f"{cell(sy.get('value') if sy.get('name') == name else None, None, sy['heldout_nrmse'])} | {closer(r)} |")
    lines += ["", "Notes:", ""]
    for r in rows:
        for key, label in (("textbook_curve_fit", "textbook"), ("pysindy_alone", "PySINDy")):
            if r[key].get("note"):
                lines.append(f"- {r['clip'].removeprefix('gallery-')}, {label}: {r[key]['note']}.")
    lines.append("- Coordinates are rebuilt from the stored track and checked by re-simulating PhysicsLens's own law on "
                 "them (`coordinates_check_nrmse` in baselines.json).")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((ROOT / "gallery").glob("gallery-*/result.json")):
        result = json.loads(path.read_text())
        if result.get("family") not in FAMILIES or "fit_state" not in result:
            print(f"{path.parent.name}: no textbook baseline for family {result.get('family')!r}; skipped")
            continue
        rows.append(run_clip(path.parent.name))
    (OUT / "baselines.json").write_text(json.dumps(rows, indent=1, default=float))
    (OUT / "BASELINES.md").write_text(markdown(rows))
    print(markdown(rows))


if __name__ == "__main__":
    main()
