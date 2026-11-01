"""Direct-track discovery for the Solar System Lab: free, fast, and never mistaken for a video discovery.

A scenario's target is "tracked" straight from the simulation, with Gaussian noise the size of a video tracker's
(scenarios.track), and enters the same fitting and verification as a clip: fit_family, rolling-origin forecasts, BIC
with the effective sample size, bootstrap intervals. The candidate laws are a fixed list instead of the reasoning
model's proposals, so it costs nothing: Kepler, a free power law, and Kepler plus a named perturber whose simulated
path is a known input. Every result says "simulation, no pixels".
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..physics.coords import Coordinates
from ..physics.custom import MAX_PARAMS_ORBIT, FreeParam, InputSignals, build, register
from ..physics.fit import fit_family
from ..physics.models import MODELS, ModelSpec, Param
from ..physics.residuals import period_matches, spectrum
from ..physics.uncertainty import bootstrap_full
from ..physics.verify import assess, judge_candidates
from .scenarios import ScenarioRun, track

SOURCE = "simulation, no pixels"
# The fitter simulates at a relative tolerance of 1e-7, which accumulates to about 1e-6 of the orbit over a few orbits
# (measured on noise-free Apophis: 1.5e-6 AU on a 1.1 AU orbit). Residuals below this floor are that precision, not
# a force; Neptune's signal on Uranus is 2.6e-4 of the orbit, thirteen times above it.
PRECISION_FLOOR = 2e-5


def kepler(gm_guess: float):
    return build("plane", "Inverse-square gravity (Kepler)", ["-mu*x/(x^2+y^2)^1.5", "-mu*y/(x^2+y^2)^1.5"],
                 [FreeParam("mu", "AU^3/day^2", gm_guess * 0.5, gm_guess * 2, gm_guess)], max_params=MAX_PARAMS_ORBIT)


def power_law(gm_guess: float):
    return build("plane", "Central force with a free exponent", ["-mu*x/(x^2+y^2)^(n/2)", "-mu*y/(x^2+y^2)^(n/2)"],
                 [FreeParam("mu", "AU^3/day^2", gm_guess * 0.5, gm_guess * 2, gm_guess),
                  FreeParam("n", "", 2.0, 4.0, 3.0)], max_params=MAX_PARAMS_ORBIT)


def kepler_plus(gm_guess: float, perturber: str, signals: InputSignals, gm_k_guess: float):
    """Kepler plus one known body's pull, with the indirect term: the body pulls the Sun too, and the frame follows."""
    x_term = "gk*((xk - x)/((xk - x)^2 + (yk - y)^2)^1.5 - xk/(xk^2 + yk^2)^1.5)"
    y_term = "gk*((yk - y)/((xk - x)^2 + (yk - y)^2)^1.5 - yk/(xk^2 + yk^2)^1.5)"
    return build("plane", f"Kepler plus {perturber}'s pull",
                 [f"-mu*x/(x^2+y^2)^1.5 + {x_term}", f"-mu*y/(x^2+y^2)^1.5 + {y_term}"],
                 [FreeParam("mu", "AU^3/day^2", gm_guess * 0.5, gm_guess * 2, gm_guess),
                  FreeParam("gk", "AU^3/day^2", 0.0, max(gm_k_guess * 100, 1e-12), gm_k_guess)],
                 signals, max_params=MAX_PARAMS_ORBIT)


def perturber_signals(sr: ScenarioRun, name: str) -> InputSignals:
    """A simulated body's path in the target's orbital plane, as the known inputs xk, yk."""
    pos = sr.run.of(name)[0][np.isin(sr.run.jd, sr.frames_jd)] @ sr.basis[:2].T
    t = sr.frames_jd - sr.frames_jd[0]
    return InputSignals(tuple(t), {"xk": tuple(pos[:, 0]), "yk": tuple(pos[:, 1])}, {"xk": "AU", "yk": "AU"},
                        {"xk": f"{name}, {SOURCE}", "yk": f"{name}, {SOURCE}"})


class KnownPull:
    """The summed pull (with the indirect term) of bodies whose paths and masses are known, in the target's plane:
    what Le Verrier subtracted before looking for Neptune. Paths are cubic splines through the simulated positions."""

    def __init__(self, sr: ScenarioRun, names: tuple[str, ...]):
        from scipy.interpolate import CubicSpline
        t = sr.frames_jd - sr.frames_jd[0]
        rows = np.isin(sr.run.jd, sr.frames_jd)
        self.names = names
        self.gm = [(_gm_of(sr, n) or 0.0) for n in names]
        self.paths = [CubicSpline(t, sr.run.of(n)[0][rows] @ sr.basis[:2].T) for n in names]

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        ax = ay = 0.0
        for gm, path in zip(self.gm, self.paths, strict=True):
            xk, yk = path(t)
            dx, dy = xk - x, yk - y
            d3 = (dx * dx + dy * dy) ** 1.5
            rk3 = (xk * xk + yk * yk) ** 1.5
            ax += gm * (dx / d3 - xk / rk3)
            ay += gm * (dy / d3 - yk / rk3)
        return ax, ay


def _none(x: float, y: float, t: float) -> tuple[float, float]:
    return 0.0, 0.0


def kepler_known(gm_guess: float, known: KnownPull) -> ModelSpec:
    """Kepler plus the known bodies' pull (their masses and paths given): one fitted parameter, mu."""
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float], t: float) -> np.ndarray:
        x, y, vx, vy = s
        r3 = (x * x + y * y) ** 1.5
        kx, ky = known(x, y, t)
        return np.array([vx, vy, -p["mu"] * x / r3 + kx, -p["mu"] * y / r3 + ky])

    spec = ModelSpec("lab_known_bodies", "plane", f"Kepler plus the known pull of {', '.join(known.names)}",
                     r"\ddot{\mathbf r} = -\mu\frac{\mathbf r}{r^3} + \sum_k G m_k\left(\frac{\mathbf r_k - \mathbf r}"
                     r"{|\mathbf r_k - \mathbf r|^3} - \frac{\mathbf r_k}{r_k^3}\right)",
                     (Param("mu", r"\mu", "AU^3/day^2", gm_guess * 0.5, gm_guess * 2, gm_guess),), rhs,
                     time_dependent=True)
    MODELS[spec.id] = spec
    return spec


def hidden_body(gm_guess: float, target_a: float, known=_none) -> ModelSpec:
    """Kepler plus an unseen body on a circular orbit in the plane: its mass gp, radius R and phase phi are fitted; its
    angular speed follows from Kepler's third law with the fitted mu, so it is not a free parameter (Le Verrier's
    problem). Heliocentric, so its pull on the Sun (the indirect term) is included."""
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float], t: float) -> np.ndarray:
        x, y, vx, vy = s
        w = math.sqrt(p["mu"] / p["R"] ** 3)
        xp, yp = p["R"] * math.cos(w * t + p["phi"]), p["R"] * math.sin(w * t + p["phi"])
        r3 = (x * x + y * y) ** 1.5
        dx, dy = xp - x, yp - y
        d3 = (dx * dx + dy * dy) ** 1.5
        rp3 = p["R"] ** 3
        kx, ky = known(x, y, t)
        return np.array([vx, vy, -p["mu"] * x / r3 + p["gp"] * (dx / d3 - xp / rp3) + kx,
                         -p["mu"] * y / r3 + p["gp"] * (dy / d3 - yp / rp3) + ky])

    spec = ModelSpec("lab_hidden_body", "plane", "Kepler plus a hidden body on a circular orbit",
                     r"\ddot{\mathbf r} = -\mu\frac{\mathbf r}{r^3} + g_p\left(\frac{\mathbf r_p - \mathbf r}"
                     r"{|\mathbf r_p - \mathbf r|^3} - \frac{\mathbf r_p}{R^3}\right),\quad"
                     r" \mathbf r_p = R(\cos(\omega t + \varphi), \sin(\omega t + \varphi)),\ \omega = \sqrt{\mu/R^3}",
                     (Param("mu", r"\mu", "AU^3/day^2", gm_guess * 0.5, gm_guess * 2, gm_guess),
                      Param("gp", "g_p", "AU^3/day^2", 0.0, gm_guess * 1e-3, gm_guess * 3e-5),
                      Param("R", "R", "AU", target_a * 1.15, target_a * 3.0, target_a * 1.6),
                      Param("phi", r"\varphi", "rad", -math.pi, math.pi, 0.0)),
                     rhs, time_dependent=True)
    MODELS[spec.id] = spec
    return spec


def along_track(gm_guess: float, known=_none) -> ModelSpec:
    """Kepler plus a small constant push along the motion (a Yarkovsky-like or drag-like term), one parameter."""
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float], t: float) -> np.ndarray:
        x, y, vx, vy = s
        r3 = (x * x + y * y) ** 1.5
        v = math.hypot(vx, vy) or 1.0
        kx, ky = known(x, y, t)
        return np.array([vx, vy, -p["mu"] * x / r3 + p["A"] * vx / v + kx, -p["mu"] * y / r3 + p["A"] * vy / v + ky])

    spec = ModelSpec("lab_along_track", "plane", "Kepler plus a constant along-track push",
                     r"\ddot{\mathbf r} = -\mu\frac{\mathbf r}{r^3} + A\,\hat{\mathbf v}",
                     (Param("mu", r"\mu", "AU^3/day^2", gm_guess * 0.5, gm_guess * 2, gm_guess),
                      Param("A", "A", "AU/day^2", -1e-6, 1e-6, 0.0)), rhs, time_dependent=True)
    MODELS[spec.id] = spec
    return spec


def _target_a(sr: ScenarioRun) -> float:
    pos, vel = sr.run.of(sr.scenario.target)
    from .elements import to_elements
    return to_elements(np.concatenate([pos[0], vel[0]]), sr.central_gm).a


def _estimates(spec: ModelSpec, c: Coordinates, fit, reps: int) -> dict[str, dict[str, float]]:
    boot = bootstrap_full(spec, c, fit, reps=reps)
    return {n: {"value": e.value, "std": e.std, "ci95": [e.ci_low, e.ci_high]} for n, e in boot.estimates.items()}


def equation_diff(before: ModelSpec, after: ModelSpec, est_before: dict, est_after: dict, f_before, f_after) -> dict:
    """How the equation changed: the laws, the added terms, each shared parameter's shift with its interval, the
    forecast error before and after, and one sentence made from those numbers (a template, not the language model)."""
    added = [n for n in after.param_names if n not in before.param_names]
    shifts = {n: {"before": est_before[n], "after": est_after[n]} for n in before.param_names if n in est_after}
    sentence = (f"Adding {after.title.lower().replace('kepler plus ', '')} ({', '.join(added) or 'no new parameters'}) "
                f"changes the held-out forecast error from {f_before.heldout_nrmse:.2%} to {f_after.heldout_nrmse:.2%}")
    if "mu" in shifts:
        b, a = shifts["mu"]["before"], shifts["mu"]["after"]
        sentence += (f", and moves the central GM from {b['value']:.6e} ± {b['std']:.1e} to {a['value']:.6e} ± "
                     f"{a['std']:.1e}")
    return {"before": {"title": before.title, "latex": before.latex}, "after": {"title": after.title, "latex": after.latex},
            "added_terms": added, "parameter_shifts": shifts, "heldout_before": f_before.heldout_nrmse,
            "heldout_after": f_after.heldout_nrmse, "summary": sentence + "."}


def discover(sr: ScenarioRun, noise_px: float = 2.0, perturbers: tuple[str, ...] = (), seed: int = 0,
             bootstrap_reps: int = 24, look_for_perturbations: bool = True) -> dict[str, Any]:
    """Fit Kepler (and a free power law) to the target's simulated track; when Kepler is accepted but its residuals
    are structured, try perturbing laws and let verification decide. Reported as a result would be."""
    t, xy, px_per_au = track(sr, noise_px, seed=seed)
    c = Coordinates("plane", t, xy, ("x", "y"), {})
    gm = sr.central_gm
    # Bodies in the scenario that are not hidden are known, with their masses and paths (as Jupiter and Saturn were
    # to Le Verrier): the first law to test is Kepler plus their pull. Hidden bodies are what is left to discover.
    known_names = tuple(n for n in sr.run.names if n != sr.scenario.target and n not in sr.scenario.hidden
                        and (_gm_of(sr, n) or 0.0) > 0)
    known = KnownPull(sr, known_names) if known_names else _none
    first = [register(kepler(gm)), register(power_law(gm))] + ([kepler_known(gm, known)] if known_names else [])
    fits = fit_family(first, c)
    verdict = assess(c, fits)
    out: dict[str, Any] = {"source": SOURCE, "noise_px": noise_px, "px_per_au": px_per_au}
    trail = [{"model": k, "status": v[0], "why": v[1]} for k, v in judge_candidates(verdict).items()]
    final, specs = verdict, list(first)
    if look_for_perturbations and verdict.status == "accepted" and verdict.winner is not None:
        lead = verdict.winner
        vel = np.gradient(lead.sim, t, axis=0)
        spec_res = spectrum(t, lead.sim, vel, xy - lead.sim)
        periods = _catalogue_periods(sr)
        target_period = 2 * math.pi * math.sqrt(_target_a(sr) ** 3 / gm)
        spec_res.matches = period_matches(spec_res.peaks, target_period, periods)
        # Noise-free residuals are the fitter's own numerical precision, perfectly structured and meaningless: a
        # residual points at a missing force only above PRECISION_FLOOR. Without this, zero noise "found" a planet.
        floor = PRECISION_FLOOR * float(np.max(np.linalg.norm(xy, axis=1)))
        if spec_res.significant and max(spec_res.rms.values()) < floor:
            spec_res.significant = False
        out["residual_spectrum"] = spec_res.summary()
        if spec_res.significant:
            extra = [register(kepler_plus(gm, n, perturber_signals(sr, n), _gm_of(sr, n) or gm * 1e-3))
                     for n in perturbers] + [hidden_body(gm, _target_a(sr), known), along_track(gm, known)]
            more = []
            for sp in extra:
                starts = tuple({**{q.name: q.guess for q in sp.params if q.guess is not None}, "phi": ph}
                               for ph in (-2.4, -0.8, 0.8, 2.4)) if "phi" in sp.param_names else ()
                from ..physics.fit import fit_model
                more.append(fit_model(sp, c, starts=starts))
            final = assess(c, [*verdict.ranked, *more])
            specs += extra
            trail += [{"model": k, "status": v[0], "why": v[1]} for k, v in judge_candidates(final).items()
                      if k in {s.id for s in extra}]
            if final.winner is not None and final.winner.model_id != lead.model_id:
                before = next(s for s in specs if s.id == lead.model_id)
                after = next(s for s in specs if s.id == final.winner.model_id)
                out["equation_diff"] = equation_diff(before, after, _estimates(before, c, lead, bootstrap_reps),
                                                     _estimates(after, c, final.winner, bootstrap_reps), lead, final.winner)
            else:
                out["perturbation"] = "The residuals are structured, but no perturbing law forecasts better: the law stands."
        else:
            out["perturbation"] = "No perturbation needed: the residuals look like noise."
    elif look_for_perturbations:
        out["perturbation"] = ("No single law is singled out at this noise, so there is no accepted law to look for "
                               "perturbations in: the data cannot show them"
                               + (f" ({sr.detectability.message})" if sr.detectability else "."))
    judged = judge_candidates(final)
    out.update({"status": final.status, "reason": final.reason, "n_eff": final.n_eff, "trail": trail,
                "ranking": [{"title": next(s.title for s in specs if s.id == f.model_id), "params": f.params,
                             "heldout_nrmse": f.heldout_nrmse, "bic": f.bic,
                             "status": judged.get(f.model_id, ("", ""))[0], "why": judged.get(f.model_id, ("", ""))[1]}
                            for f in final.ranked]})
    if final.winner is not None:
        spec = next(s for s in specs if s.id == final.winner.model_id)
        boot = bootstrap_full(spec, c, final.winner, reps=bootstrap_reps)
        out["law"] = {"title": spec.title, "latex": spec.latex}
        out["parameters"] = [{"name": n, "value": e.value, "std": e.std, "ci95": [e.ci_low, e.ci_high]}
                             for n, e in boot.estimates.items()]
        out["bootstrap_reps"] = boot.reps
    return out


def _catalogue_periods(sr: ScenarioRun) -> dict[str, float]:
    """Orbital periods (days) of the scenario's other bodies, from their states at the start."""
    from .elements import to_elements
    out = {}
    for i, name in enumerate(sr.run.names):
        if name == sr.scenario.target:
            continue
        try:
            el = to_elements(np.concatenate([sr.run.pos[0, i], sr.run.vel[0, i]]), sr.central_gm)
        except ValueError:
            continue
        out[name] = el.period
    return out


def _gm_of(sr: ScenarioRun, name: str) -> float | None:
    from .ephem import masses_km3_s2
    from .nbody import gm_au3_d2
    gms = masses_km3_s2()
    return gm_au3_d2(gms[name]) * sr.scenario.mass_scale.get(name, 1.0) if name in gms else None
