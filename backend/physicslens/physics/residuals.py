"""What an accepted law leaves behind: is something else pulling?

Residuals of an orbit fit are split into the orbit's own frame (radial, along-track, cross-track; cross-track is zero
for a 2D track, the code stays general), and each is searched for periodic structure with a Lomb-Scargle periodogram
and for a secular drift with a straight-line fit. Peaks are matched with periods of known bodies (their own periods,
half periods, and synodic periods with the target) as hints for the agent, never as verdicts: only fitting and
held-out verification accept an extra term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lombscargle

FAP_SIGNIFICANT = 1e-3  # a peak this unlikely from noise alone asks for a perturbing law
TREND_SIGMA = 5.0  # a secular along-track drift this many standard errors from zero does too
MAX_PEAKS = 3


@dataclass(frozen=True)
class Peak:
    component: str
    period: float
    power: float  # normalised Lomb-Scargle power (0 to 1)
    fap: float  # false-alarm probability (Baluev-style approximation over the independent frequencies)
    amplitude: float


@dataclass
class ResidualSpectrum:
    peaks: list[Peak]
    trend: float  # along-track drift per unit time
    trend_sigma: float  # its standard error
    rms: dict[str, float]
    significant: bool
    matches: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        """Compact JSON for the agent and the UI."""
        return {"significant": self.significant, "rms": {k: float(v) for k, v in self.rms.items()},
                "peaks": [{"component": p.component, "period": round(p.period, 4), "power": round(p.power, 3),
                           "false_alarm_probability": float(f"{p.fap:.2e}"), "amplitude": float(f"{p.amplitude:.3e}")}
                          for p in self.peaks],
                "along_track_trend": float(f"{self.trend:.3e}"), "trend_sigma": float(f"{self.trend_sigma:.3e}"),
                "period_matches": self.matches}


def orbit_frame(pos: np.ndarray, vel: np.ndarray, resid: np.ndarray) -> dict[str, np.ndarray]:
    """Residual vectors (measured minus model) as radial, along-track and cross-track components."""
    p3 = np.zeros((len(pos), 3))
    v3 = np.zeros((len(pos), 3))
    r3 = np.zeros((len(pos), 3))
    p3[:, : pos.shape[1]], v3[:, : vel.shape[1]], r3[:, : resid.shape[1]] = pos, vel, resid
    radial = p3 / np.linalg.norm(p3, axis=1, keepdims=True)
    h = np.cross(p3, v3)
    cross = h / np.linalg.norm(h, axis=1, keepdims=True)
    along = np.cross(cross, radial)
    return {"radial": np.sum(r3 * radial, axis=1), "along": np.sum(r3 * along, axis=1),
            "cross": np.sum(r3 * cross, axis=1)}


def _fap(power: float, n_freq: int, n: int) -> float:
    """False-alarm probability of the highest normalised peak among n_freq independent frequencies."""
    single = (1 - power) ** ((n - 3) / 2) if power < 1 else 0.0
    return float(min(1.0, n_freq * single))


def spectrum(t: np.ndarray, pos: np.ndarray, vel: np.ndarray, resid: np.ndarray) -> ResidualSpectrum:
    """Peaks and drift of an accepted fit's residuals."""
    t = np.asarray(t, dtype=float)
    comps = orbit_frame(pos, vel, resid)
    span = float(t[-1] - t[0])
    n = len(t)
    # Periods from twice the sampling step up to the whole span, on a grid four times finer than independent.
    dt = float(np.median(np.diff(t)))
    freqs = np.linspace(1.0 / span, 0.5 / dt, max(64, 4 * n))
    n_independent = max(1, int(span / dt / 2))
    peaks: list[Peak] = []
    rms = {}
    for name, x in comps.items():
        rms[name] = float(np.sqrt(np.mean(x**2)))
        if rms[name] == 0 or np.ptp(x) == 0:
            continue
        y = x - x.mean()
        power = lombscargle(t, y, 2 * math.pi * freqs, normalize=True)  # y is already centred
        for k in np.argsort(power)[::-1]:
            if len([p for p in peaks if p.component == name]) >= MAX_PEAKS:
                break
            # A peak, not the shoulder of one already kept.
            if any(p.component == name and abs(1 / p.period - freqs[k]) < 2 / span for p in peaks):
                continue
            amp = math.sqrt(4 * power[k] * float(np.var(y)) / 2)
            peaks.append(Peak(name, 1.0 / freqs[k], float(power[k]), _fap(float(power[k]), n_independent, n), amp))
    peaks.sort(key=lambda p: p.fap)
    a = comps["along"]
    A = np.column_stack([np.ones(n), t - t.mean()])
    coef, res, *_ = np.linalg.lstsq(A, a, rcond=None)
    sigma2 = float(res[0] / max(n - 2, 1)) if len(res) else float(np.var(a - A @ coef))
    trend_sigma = math.sqrt(sigma2 / float(np.sum((t - t.mean()) ** 2)))
    significant = (bool(peaks) and peaks[0].fap < FAP_SIGNIFICANT) or abs(coef[1]) > TREND_SIGMA * trend_sigma
    return ResidualSpectrum(peaks[: MAX_PEAKS * 3], float(coef[1]), trend_sigma, rms, significant)


def period_matches(peaks: list[Peak], target_period: float, bodies: dict[str, float],
                   max_rel_error: float = 0.15) -> list[dict[str, object]]:
    """Known periods near each significant peak: a body's own period, its half, and its synodic period with the
    target (1/P_syn = |1/P_target - 1/P_body|). Ranked by relative error; hints, never verdicts."""
    out = []
    for p in peaks:
        if p.fap >= FAP_SIGNIFICANT:
            continue
        for name, period in bodies.items():
            cands = {"own period": period, "half period": period / 2}
            if abs(1 / target_period - 1 / period) > 0:
                syn = 1 / abs(1 / target_period - 1 / period)
                cands["synodic period"] = syn
                cands["half synodic period"] = syn / 2
            for what, value in cands.items():
                err = abs(p.period - value) / value
                if err <= max_rel_error:
                    out.append({"peak_period": round(p.period, 3), "component": p.component, "body": name,
                                "kind": what, "period": round(value, 3), "relative_error": round(err, 3)})
    return sorted(out, key=lambda m: m["relative_error"])
