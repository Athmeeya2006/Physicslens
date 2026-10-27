"""Machine-health drift: compare the dynamics of two recordings of the same system.

Both recordings are fitted with the same damped model and reduced to two health metrics, natural
frequency and amplitude decay rate, each with a bootstrap uncertainty. A change is flagged only when it is
both statistically clear and large enough to matter, so re-recording noise never raises an alarm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..sigproc.trajectory import Trajectory
from .coords import to_coordinates
from .fit import fit_model
from .models import MODELS
from .uncertainty import bootstrap_samples

# The damped model of each family, so damping is always measured (zero damping is a valid answer).
DRIFT_MODEL = {"pendulum": "pendulum_linear_damping", "oscillator": "damped_harmonic"}
Z_THRESHOLD = 3.0  # standard errors
# A change must also be big enough to matter. Frequency is measured very precisely (and needs no scale
# calibration), so 0.5% is meaningful; damping is harder to pin down, so it needs 10%.
MIN_RELATIVE_CHANGE = {"natural_freq_hz": 0.005, "decay_rate_per_s": 0.10}
# One clip's bootstrap only sees tracking noise. Re-recording adds release, temperature and mounting
# variation, so uncertainties never go below these repeatability floors (conservative assumptions).
REPEATABILITY = {"natural_freq_hz": ("relative", 0.002), "decay_rate_per_s": ("absolute", 0.002)}
DECAY_FLOOR = 0.005  # 1/s: below this, damping is "none measurable" and relative changes are meaningless


class HealthError(ValueError):
    pass


@dataclass(frozen=True)
class Metric:
    value: float
    std: float


@dataclass(frozen=True)
class HealthMetrics:
    family: str
    model_id: str
    natural_freq_hz: Metric
    decay_rate_per_s: Metric  # amplitude falls as exp(-rate * t)
    duration_s: float


def _floored(name: str, value: float, std: float) -> float:
    kind, amount = REPEATABILITY[name]
    return max(std, amount * abs(value) if kind == "relative" else amount)


def health_metrics(traj: Trajectory, family: str, reps: int = 20) -> HealthMetrics:
    if family not in DRIFT_MODEL:
        raise HealthError(f"Drift comparison needs an oscillating system (pendulum or spring), not {family}.")
    spec = MODELS[DRIFT_MODEL[family]]
    c = to_coordinates(traj, family)
    fit = fit_model(spec, c)
    if not fit.success:
        raise HealthError("The damped model could not be fitted to this recording.")
    samples, lengths = bootstrap_samples(spec, c, fit, reps=reps)
    idx = {name: i for i, name in enumerate(spec.param_names)}
    if family == "pendulum":
        # sqrt(g/L) per sample: L and g move together in the bootstrap, so the frequency spread stays honest.
        omega = np.sqrt(samples[:, idx["g"]] / lengths)
        decay = samples[:, idx["gamma"]] / 2
        omega_hat = np.sqrt(fit.params["g"] / c.ctx["length_m"])
        decay_hat = fit.params["gamma"] / 2
    else:
        omega = samples[:, idx["omega"]]
        decay = samples[:, idx["zeta"]] * samples[:, idx["omega"]]
        omega_hat = fit.params["omega"]
        decay_hat = fit.params["zeta"] * fit.params["omega"]
    f_hat = float(omega_hat / (2 * np.pi))
    return HealthMetrics(family, spec.id,
                         Metric(f_hat, _floored("natural_freq_hz", f_hat, float(np.std(omega / (2 * np.pi), ddof=1)))),
                         Metric(float(decay_hat), _floored("decay_rate_per_s", decay_hat, float(np.std(decay, ddof=1)))),
                         float(traj.t[-1] - traj.t[0]))


def compare(baseline: HealthMetrics, current: HealthMetrics) -> dict[str, Any]:
    if baseline.family != current.family:
        raise HealthError(f"Can't compare a {baseline.family} with a {current.family}.")
    rows = []
    for name, label, unit, a, b in [
        ("natural_freq_hz", "Natural frequency", "Hz", baseline.natural_freq_hz, current.natural_freq_hz),
        ("decay_rate_per_s", "Damping (amplitude decay rate)", "1/s", baseline.decay_rate_per_s, current.decay_rate_per_s),
    ]:
        diff = b.value - a.value
        se = float(np.hypot(a.std, b.std)) or 1e-12
        z = diff / se
        ref = max(abs(a.value), DECAY_FLOOR if name == "decay_rate_per_s" else 1e-12)
        rel = diff / ref
        flagged = abs(z) > Z_THRESHOLD and abs(rel) > MIN_RELATIVE_CHANGE[name]
        rows.append({"metric": name, "label": label, "unit": unit, "baseline": a.value, "baseline_std": a.std,
                     "current": b.value, "current_std": b.std, "change_pct": 100 * rel, "z": z, "flagged": flagged})
    changed = [r for r in rows if r["flagged"]]
    if changed:
        summary = "Dynamics drifted: " + "; ".join(
            f"{r['label'].split(' (')[0].lower()} {'up' if r['change_pct'] > 0 else 'down'} {abs(r['change_pct']):.0f}%"
            f" ({abs(r['z']):.1f} standard errors)" for r in changed) + "."
    else:
        summary = "No significant change: both recordings show the same dynamics within their uncertainty."
    return {"family": baseline.family, "model_id": baseline.model_id, "drift": bool(changed), "summary": summary,
            "metrics": rows, "durations_s": [baseline.duration_s, current.duration_s]}
