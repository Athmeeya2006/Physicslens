"""Rolling-bearing faults from physics: where each defect must ring, and whether it does.

A defect on a rolling bearing is struck at a rate fixed by kinematics: every ball passing an outer-race pit (BPFO),
an inner-race pit turning past the balls (BPFI), a ball pit touching both races (BSF), a broken cage turning (FTF).
Each impact excites the structure's resonances, so the defect rate shows up as the repetition rate of bursts, not as
a tone: the standard way to see it is the envelope spectrum of a band where the bursts dominate (Randall and Antoni
2011, "Rolling element bearing diagnostics: a tutorial", Mechanical Systems and Signal Processing 25(2) 485 to 520).
The fault type is then whichever expected rate stands out, decided by a rule written before any labelled data was
scored. No training, no learned weights: a fault nobody has labelled is found the same way.

Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis

FAULTS = ("outer", "inner", "ball", "cage")

# The decision rule, fixed before scoring (see DIAGNOSIS_RULE for the words).
HARMONICS = 3  # orders 1..3 of each defect rate
SNR_DB = 6.0  # a harmonic "stands out" when its peak is this far above the local background
MIN_HARMONICS = 2  # a defect series is present when at least this many of its harmonics stand out
REL_WINDOW = 0.01  # search +-1% around each expected line: real balls skid, and the shaft rate is known to ~0.5%
MIN_WINDOW_HZ = 0.5
BACKGROUND_HZ = 10.0  # local background: median of the envelope spectrum within this distance, window excluded
SHAFT_MASK_HZ = 0.3  # envelope lines this close to a shaft harmonic are unbalance/misalignment, not a defect
DIAGNOSIS_RULE = (
    f"Envelope spectrum of the band with the highest kurtosis. For each defect rate (BPFO, BPFI, BSF, FTF) take "
    f"harmonics 1 to {HARMONICS}; a harmonic stands out when the peak within +-{REL_WINDOW:.0%} (at least "
    f"+-{MIN_WINDOW_HZ} Hz) of the expected line, ignoring bins within {SHAFT_MASK_HZ} Hz of a shaft harmonic, is "
    f"at least {SNR_DB:g} dB above the median of the spectrum within {BACKGROUND_HZ:g} Hz. A defect is present when "
    f"at least {MIN_HARMONICS} of its {HARMONICS} harmonics stand out; the diagnosis is the present defect with the "
    f"highest mean peak level over its harmonics, or 'none' when no defect is present."
)


@dataclass(frozen=True)
class BearingGeometry:
    n_balls: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0


def defect_frequencies(geom: BearingGeometry, shaft_hz: float) -> dict[str, float]:
    """Kinematic defect rates for a stationary outer race (Randall and Antoni 2011, eqs. 1 to 4).

    BSF here is the ball spin rate; a ball defect touches both races per spin, so its envelope often peaks at 2x BSF,
    which is the second harmonic and is covered by the harmonic search.
    """
    r = geom.ball_diameter_mm / geom.pitch_diameter_mm * np.cos(np.radians(geom.contact_angle_deg))
    n = geom.n_balls
    return {
        "outer": n * shaft_hz / 2 * (1 - r),
        "inner": n * shaft_hz / 2 * (1 + r),
        "ball": geom.pitch_diameter_mm / (2 * geom.ball_diameter_mm) * shaft_hz * (1 - r**2),
        "cage": shaft_hz / 2 * (1 - r),
    }


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def kurtosis_band(x: np.ndarray, fs: float, widths_hz: tuple[float, ...] = (1000.0, 2000.0, 4000.0),
                  lo_hz: float = 500.0) -> tuple[float, float]:
    """The band where the signal is most impulsive (a coarse kurtogram, Antoni 2006).

    Defect impacts are short bursts riding on a resonance; kurtosis is high where they dominate and near zero where
    steady noise or tones do, so this picks the band without knowing the resonance in advance.
    """
    top = 0.45 * fs
    best = (-np.inf, lo_hz, lo_hz + widths_hz[0])
    for w in widths_hz:
        lo = lo_hz
        while lo + w <= top:
            k = float(kurtosis(_bandpass(x, fs, lo, lo + w), fisher=True))
            if k > best[0]:
                best = (k, lo, lo + w)
            lo += w / 2
    return best[1], best[2]


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float],
                      resolution_hz: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Welch spectrum of the squared envelope: averaging segments keeps the background steady for a fixed dB rule."""
    env = np.abs(hilbert(_bandpass(x, fs, *band))) ** 2
    env = env - env.mean()
    nper = min(len(env), round(fs / resolution_hz))
    f, p = welch(env, fs=fs, nperseg=nper, noverlap=3 * nper // 4, window="hann", detrend=False)
    return f, p


def shaft_rate_from_spectrum(x: np.ndarray, fs: float, nominal_hz: float, tolerance: float = 0.03) -> float | None:
    """The 1x line of the raw vibration near the tachometer value; None when no clear peak (then trust the tacho)."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 4)), noverlap=int(fs * 3), detrend="constant")
    sel = (f > nominal_hz * (1 - tolerance)) & (f < nominal_hz * (1 + tolerance))
    near = (f > nominal_hz * 0.7) & (f < nominal_hz * 1.3) & ~sel
    if not sel.any() or not near.any():
        return None
    i = np.argmax(np.where(sel, p, -np.inf))
    if 10 * np.log10(p[i] / np.median(p[near])) < SNR_DB:
        return None
    return float(f[i])


def line_snr_db(f: np.ndarray, p: np.ndarray, target_hz: float, shaft_hz: float) -> float:
    """Peak near target over the local background, in dB, with shaft-harmonic bins removed from the search."""
    half = max(MIN_WINDOW_HZ, REL_WINDOW * target_hz)
    win = np.abs(f - target_hz) <= half
    order = np.round(f / shaft_hz)
    shaft_line = (order >= 1) & (np.abs(f - order * shaft_hz) <= SHAFT_MASK_HZ)
    search = win & ~shaft_line
    local = (np.abs(f - target_hz) <= BACKGROUND_HZ) & ~win
    if not search.any() or not local.any():
        return float("nan")
    bg = np.median(p[local])
    return float(10 * np.log10(p[search].max() / bg)) if bg > 0 else float("nan")


@dataclass(frozen=True)
class Diagnosis:
    fault: str  # one of FAULTS, or "none"
    shaft_hz: float
    band_hz: tuple[float, float]
    expected_hz: dict[str, float]
    snr_db: dict[str, list[float]]  # per defect, harmonics 1..HARMONICS
    present: dict[str, bool]


def diagnose(x: np.ndarray, fs: float, geom: BearingGeometry, tach_shaft_hz: float,
             band: tuple[float, float] | None = None) -> Diagnosis:
    measured = shaft_rate_from_spectrum(x, fs, tach_shaft_hz)
    shaft = measured if measured is not None else tach_shaft_hz
    band = band or kurtosis_band(x, fs)
    f, p = envelope_spectrum(x, fs, band)
    expected = defect_frequencies(geom, shaft)
    snr = {k: [line_snr_db(f, p, h * v, shaft) for h in range(1, HARMONICS + 1)] for k, v in expected.items()}
    present = {k: sum(s >= SNR_DB for s in v if np.isfinite(s)) >= MIN_HARMONICS for k, v in snr.items()}
    candidates = [k for k in FAULTS if present[k]]
    fault = max(candidates, key=lambda k: float(np.nanmean(snr[k]))) if candidates else "none"
    return Diagnosis(fault, shaft, band, expected, snr, present)


# ---------------------------------------------------------------- second rule: against the same bearing's own baseline
#
# Written after the first rule failed on UORED-VAFCLS, for two physical reasons found while checking the measurements
# (not by tuning thresholds, which are unchanged): the single Hall-sensor speed stored per file disagrees with the
# 1x vibration line by up to 23% (2190 rpm logged, 1785 measured), and in most files the 1x line sits just below
# 30 Hz, the synchronous speed of a 4-pole motor on 60 Hz; and the test motor's own internal bearings have the same
# geometry, so a healthy recording can already carry a defect series. A fault is therefore read as a defect series
# that GREW against the same bearing's healthy recording, with the shaft rate measured from the vibration within the
# induction motor's physical range. It did not do better (see data/health_public/REPORT.md).
GROWTH_DB = SNR_DB  # same 6 dB, now as growth over the bearing's own healthy level
BASELINE_RULE = (
    f"Shaft rate: the 1x vibration line within the motor's slip range (synchronous speed down to 10% slip), "
    f"interpolated. Envelope spectra of the test recording and of the same bearing's healthy recording, both in the "
    f"test recording's highest-kurtosis band. A defect series grew when at least {MIN_HARMONICS} of its "
    f"{HARMONICS} harmonics both stand out in the test recording (at least {SNR_DB:g} dB over local background) and "
    f"rose at least {GROWTH_DB:g} dB over the healthy recording; the diagnosis is the grown series with the largest "
    f"mean rise, or 'none'."
)


def shaft_rate_in_range(x: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> float | None:
    """Strongest vibration line in [lo, hi], parabolic-interpolated on the full-length spectrum (0.1 Hz bins for 10 s).

    An induction motor turns a little below its synchronous speed (slip), so this range is set by the motor's
    physics, not by a tachometer reading.
    """
    n = len(x)
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    sel = np.flatnonzero((f >= lo_hz) & (f <= hi_hz))
    near = (f > lo_hz * 0.7) & (f < hi_hz * 1.3) & ~((f >= lo_hz) & (f <= hi_hz))
    i = sel[np.argmax(spec[sel])]
    if 10 * np.log10(spec[i] / np.median(spec[near])) < SNR_DB or i in (0, len(f) - 1):
        return None
    a, b, c = np.log(spec[i - 1:i + 2])
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float(f[i] + shift * (f[1] - f[0]))


@dataclass(frozen=True)
class BaselineDiagnosis:
    fault: str
    shaft_hz: float
    baseline_shaft_hz: float
    band_hz: tuple[float, float]
    snr_db: dict[str, list[float]]
    growth_db: dict[str, list[float]]
    grown: dict[str, bool]


def _series(x: np.ndarray, fs: float, geom: BearingGeometry, shaft: float,
            band: tuple[float, float]) -> dict[str, list[float]]:
    f, p = envelope_spectrum(x, fs, band)
    return {k: [line_snr_db(f, p, h * v, shaft) for h in range(1, HARMONICS + 1)]
            for k, v in defect_frequencies(geom, shaft).items()}


def diagnose_against_baseline(x: np.ndarray, baseline: np.ndarray, fs: float, geom: BearingGeometry,
                              shaft_range_hz: tuple[float, float], fallback_shaft_hz: float) -> BaselineDiagnosis:
    shaft = shaft_rate_in_range(x, fs, *shaft_range_hz) or fallback_shaft_hz
    base_shaft = shaft_rate_in_range(baseline, fs, *shaft_range_hz) or fallback_shaft_hz
    band = kurtosis_band(x, fs)
    test = _series(x, fs, geom, shaft, band)
    base = _series(baseline, fs, geom, base_shaft, band)
    growth = {k: [t - b for t, b in zip(test[k], base[k], strict=True)] for k in test}
    grown = {k: sum(t >= SNR_DB and g >= GROWTH_DB for t, g in zip(test[k], growth[k], strict=True)
                    if np.isfinite(t) and np.isfinite(g)) >= MIN_HARMONICS for k in test}
    candidates = [k for k in FAULTS if grown[k]]
    fault = max(candidates, key=lambda k: float(np.nanmean(growth[k]))) if candidates else "none"
    return BaselineDiagnosis(fault, shaft, base_shaft, band, test, growth, grown)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: stays inside [0, 1] and is honest at the small counts a public dataset gives."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def synthetic_fault(fs: float, seconds: float, defect_hz: float, resonance_hz: float = 3000.0,
                    shaft_hz: float = 29.0, impact: float = 1.0, noise: float = 0.5, modulate_shaft: bool = False,
                    seed: int = 0) -> np.ndarray:
    """Decaying resonance bursts repeating at defect_hz (with 1% jitter, as skidding balls give) in white noise.

    With modulate_shaft the burst size follows the shaft angle, as an inner-race pit moving through the load zone.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = noise * rng.standard_normal(n)
    if impact > 0 and defect_hz > 0:
        ring = np.exp(-800 * t[: int(fs * 0.01)]) * np.sin(2 * np.pi * resonance_hz * t[: int(fs * 0.01)])
        period = 1 / defect_hz
        tk = period * rng.uniform(0, 1)
        while tk < seconds:
            i = int(tk * fs)
            a = impact * ((1 + np.cos(2 * np.pi * shaft_hz * tk)) / 2 if modulate_shaft else 1.0)
            seg = ring[: n - i]
            x[i:i + len(seg)] += a * seg
            tk += period * (1 + 0.01 * rng.standard_normal())
    x += 0.3 * np.sin(2 * np.pi * shaft_hz * t)  # residual unbalance: a shaft tone every machine has
    return x
