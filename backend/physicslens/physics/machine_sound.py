"""Anomalous machine sound from physics: rotation rate and harmonic structure, compared with the machine's own normal.

A rotating machine (fan, pump) sings at its rotation rate and its harmonics; blade or vane passing puts more energy
at a fixed multiple. Unbalance raises the 1x line, a change of supply voltage or load shifts the rate, clogging and
leaks add broadband flow noise. So a small set of physical quantities, each measured against that machine's normal
recordings, is a detector that says what moved, not only that something did. Scores are robust z-scores against
normal clips only: no anomalous example is ever used to set anything.

Pure functions, no I/O.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import welch
from scipy.stats import norm

RESOLUTION_HZ = 0.25
BACKGROUND_HZ = 10.0  # running-median width for the "excess over background" spectrum
F0_RANGE_HZ = (15.0, 100.0)  # rotation of mains-driven fans and pumps (900 to 6000 rpm)
N_HARMONICS = 8
TRACK = 0.05  # a clip's rotation rate is searched within +-5% of the machine's normal rate
FEATURES = tuple(["rotation_hz"] + [f"h{k}_db" for k in range(1, N_HARMONICS + 1)]
                 + ["floor_low_db", "floor_high_db"])


@dataclass(frozen=True)
class Spectrum:
    f: np.ndarray
    level_db: np.ndarray  # power spectral density, dB
    excess_db: np.ndarray  # level minus its running median: tonal lines stand out, broadband noise sits near 0


def spectrum(x: np.ndarray, fs: float) -> Spectrum:
    nper = min(len(x), round(fs / RESOLUTION_HZ))
    f, p = welch(x.astype(float), fs=fs, nperseg=nper, noverlap=nper // 2, detrend="constant")
    lp = 10 * np.log10(np.maximum(p, 1e-30))
    width = int(BACKGROUND_HZ / (f[1] - f[0])) | 1
    return Spectrum(f, lp, lp - median_filter(lp, size=width, mode="nearest"))


def _line(s: Spectrum, target: float) -> float:
    half = max(0.5, 0.005 * target)
    sel = np.abs(s.f - target) <= half
    return float(s.excess_db[sel].max()) if sel.any() else float("nan")


def harmonic_salience(s: Spectrum, f0: float, k: int = N_HARMONICS) -> float:
    return float(np.nanmean([_line(s, h * f0) for h in range(1, k + 1)]))


def reference_rotation(refs: list[Spectrum], step_hz: float = 0.05) -> float:
    """The rate whose harmonic comb best fits the machine's normal clips, averaged so one noisy clip cannot decide.

    This can land on the rotation rate or on the blade-passing rate (a fixed multiple of it); either is fine here,
    because both move together when the speed changes and the clip-by-clip search stays near the same comb.
    """
    grid = np.arange(F0_RANGE_HZ[0], F0_RANGE_HZ[1], step_hz)
    score = np.array([np.mean([harmonic_salience(s, g) for s in refs]) for g in grid])
    return float(grid[int(np.argmax(score))])


def clip_features(s: Spectrum, f_ref: float, step_hz: float = 0.02) -> dict[str, float]:
    grid = np.arange(f_ref * (1 - TRACK), f_ref * (1 + TRACK), step_hz)
    f0 = float(grid[int(np.argmax([harmonic_salience(s, g) for g in grid]))])
    feats = {"rotation_hz": f0}
    for k in range(1, N_HARMONICS + 1):
        feats[f"h{k}_db"] = _line(s, k * f0)
    # Broadband floors: the median level ignores tonal lines, so these follow flow and friction noise.
    feats["floor_low_db"] = float(np.median(s.level_db[(s.f >= 50) & (s.f < 1000)]))
    feats["floor_high_db"] = float(np.median(s.level_db[(s.f >= 1000) & (s.f < 8000)]))
    return feats


def robust_reference(rows: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Median and MAD-based spread of each feature over normal clips (one odd normal clip cannot move them)."""
    out = {}
    for k in FEATURES:
        v = np.array([r[k] for r in rows], dtype=float)
        med = float(np.nanmedian(v))
        mad = 1.4826 * float(np.nanmedian(np.abs(v - med)))
        out[k] = (med, max(mad, 1e-6 if k != "rotation_hz" else 0.01))
    return out


def z_scores(row: dict[str, float], ref: dict[str, tuple[float, float]]) -> dict[str, float]:
    return {k: (row[k] - ref[k][0]) / ref[k][1] for k in FEATURES}


def anomaly_score(row: dict[str, float], ref: dict[str, tuple[float, float]]) -> float:
    """Root mean square of the robust z-scores: every physical quantity counts equally, fixed before scoring."""
    z = np.array(list(z_scores(row, ref).values()))
    return float(np.sqrt(np.nanmean(z**2)))


def auc_delong(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    """AUC with a 95% interval by DeLong's variance (DeLong, DeLong and Clarke-Pearson 1988)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = float(psi.mean())
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    var = np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg)
    half = norm.ppf(0.975) * np.sqrt(var)
    return auc, float(max(0.0, auc - half)), float(min(1.0, auc + half))


def band_levels(x: np.ndarray, fs: float, n_bands: int = 64) -> np.ndarray:
    """Log power in log-spaced bands, for the generic (not physics) comparison detector."""
    s = spectrum(x, fs)
    edges = np.geomspace(20, fs / 2 * 0.98, n_bands + 1)
    p = 10 ** (s.level_db / 10)
    return np.array([10 * np.log10(p[(s.f >= a) & (s.f < b)].mean() + 1e-30) for a, b in itertools.pairwise(edges)])
