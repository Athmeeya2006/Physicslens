"""Experiment design: when candidate equations tie, find the cheapest new recording that would separate them.

Each tied model, with its own fitted parameters, is simulated under a menu of feasible experiments (a larger
release angle, a longer recording, a higher drop, a faster spin). The experiment is useful when the models'
predicted motions differ by several times the tracking noise measured on this clip; the least demanding
useful one is recommended. Pure simulation: no data is invented, only the next measurement is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from .coords import Coordinates
from .fit import FitResult, simulate
from .models import MODELS

SEPARATION_TARGET = 5.0  # predicted difference / tracking noise needed to call an experiment decisive
FPS = 30.0


@dataclass(frozen=True)
class Option:
    label: str  # plain-language instruction for the user
    seconds: float
    effort: float  # for ranking: lower is easier to do
    y0: dict[str, np.ndarray]  # initial state per model id (models may differ in what a "release" means)


def _pendulum_options(c: Coordinates, fits: list[FitResult], duration: float) -> list[Option]:
    amp_now = float(np.max(np.abs(c.q[:, 0])))
    out = []
    for deg in sorted({round(np.degrees(amp_now)), 20, 35, 50}):
        for secs in sorted({round(duration), 15, 25}):
            th0 = np.radians(deg)
            label = f"release from about {deg} degrees and film {secs} s"
            out.append(Option(label, secs, deg / 10 + secs / 5, {f.model_id: np.array([th0, 0.0]) for f in fits}))
    return out


def _oscillator_options(c: Coordinates, fits: list[FitResult], duration: float) -> list[Option]:
    amp_now = float(np.ptp(c.q[:, 0]) / 2)
    out = []
    for k in (1, 2, 3):
        for secs in sorted({round(duration), 2 * round(duration), 30}):
            label = (f"pull it {k} times as far" if k > 1 else "use the same pull") + f" and film {secs} s"
            out.append(Option(label, secs, k * 2 + secs / 5,
                              {f.model_id: np.array([f.params.get("s0", 0.0) + k * amp_now, 0.0]) for f in fits}))
    return out


def _projectile_options(c: Coordinates, fits: list[FitResult], duration: float) -> list[Option]:
    # Higher drops reach higher speeds, where drag laws diverge.
    out = []
    for h in (1.0, 2.0, 4.0):
        secs = float(np.sqrt(2 * h / 9.81))
        out.append(Option(f"drop it from {h:g} m", secs, h, {f.model_id: np.array([0.0, h, 0.0, 0.0]) for f in fits}))
    return out


def _rotation_options(c: Coordinates, fits: list[FitResult], duration: float) -> list[Option]:
    w_now = float(np.max(np.abs(np.gradient(c.q[:, 0], c.t))))
    out = []
    for k in (1, 2):
        for secs in sorted({round(duration), 2 * round(duration), 20}):
            label = ("spin it twice as fast" if k == 2 else "spin it as before") + f" and film {secs} s, until it stops"
            out.append(Option(label, secs, k * 2 + secs / 5, {f.model_id: np.array([0.0, k * w_now]) for f in fits}))
    return out


_OPTIONS = {"pendulum": _pendulum_options, "oscillator": _oscillator_options,
            "projectile": _projectile_options, "rotation": _rotation_options}


def _noise_m(c: Coordinates, fits: list[FitResult]) -> float:
    """Tracking noise in meters on screen (angles are converted with the arm or string length)."""
    sigma = float(np.median([f.noise_sigma for f in fits]))
    scale = c.ctx.get("length_m") or c.ctx.get("arm_m") or 1.0
    return max(sigma * (scale if c.family in ("pendulum", "rotation") else 1.0), 1e-4)


def design_experiment(c: Coordinates, contenders: list[FitResult]) -> dict[str, Any] | None:
    """Rank feasible experiments by how decisively they would separate the tied models."""
    if len(contenders) < 2 or c.family not in _OPTIONS:
        return None
    duration = float(c.t[-1] - c.t[0])
    noise = _noise_m(c, contenders)
    scored = []
    for opt in _OPTIONS[c.family](c, contenders, duration):
        t = np.arange(0, opt.seconds, 1 / FPS)
        paths = {}
        for f in contenders:
            q = simulate(MODELS[f.model_id], f.params, opt.y0[f.model_id], t, c.ctx)
            if q is None:
                break
            paths[f.model_id] = Coordinates(c.family, t, q, c.names, c.ctx).to_xy(q)
        if len(paths) < len(contenders):
            continue
        # The hardest pair to tell apart decides whether the experiment settles the question.
        sep = min(float(np.sqrt(np.mean(np.sum((paths[a] - paths[b]) ** 2, axis=1))))
                  for a, b in combinations(paths, 2))
        scored.append({"label": opt.label, "seconds": opt.seconds, "effort": opt.effort,
                       "separation_mm": 1000 * sep, "separation_vs_noise": sep / noise})
    if not scored:
        return None
    decisive = sorted((s for s in scored if s["separation_vs_noise"] >= SEPARATION_TARGET), key=lambda s: s["effort"])
    best = decisive[0] if decisive else max(scored, key=lambda s: s["separation_vs_noise"])
    return {
        "recommended": best, "decisive": bool(decisive), "noise_mm": 1000 * noise, "target_ratio": SEPARATION_TARGET,
        "options_tested": len(scored),
        "top": sorted(scored, key=lambda s: (s["separation_vs_noise"] < SEPARATION_TARGET, s["effort"]))[:4],
        "models": [MODELS[f.model_id].title for f in contenders],
    }
