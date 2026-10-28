"""The microphone as a second sensor: impact onsets, restitution from flight times, tones and spin-down from sound.

Pure functions on a mono sample array; no files, no network. The physics laws are not restated here: a spin-down
heard as a falling tone is turned into a phase angle and fitted with the same spin models the camera uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt, spectrogram

from ..physics.coords import Coordinates
from ..physics.fit import fit_model
from ..physics.models import MODELS
from ..physics.uncertainty import bootstrap
from ..physics.verify import assess

SAMPLE_RATE = 48_000
SPIN_LAWS = ("spin_free", "spin_dry", "spin_viscous", "spin_air", "spin_dry_viscous")
AGREE_SIGMA = 3.0
MAX_E_JUMP = 0.15  # restitution changes with impact speed, but not by this much between neighbouring bounces


class AcousticError(ValueError):
    """The sound does not contain what was asked for (no bounce train, no clear tone); not a crash."""


@dataclass(frozen=True)
class Onsets:
    times: np.ndarray  # seconds from the start of the audio
    strength: np.ndarray  # envelope peak of each event, arbitrary units
    sigma_t: float  # timing resolution of a single onset, seconds


def detect_onsets(x: np.ndarray, sr: int = SAMPLE_RATE, *, highpass_hz: float = 1000.0, smooth_s: float = 0.001,
                  min_gap_s: float = 0.03, background_s: float = 0.5, k: float = 6.0, floor_rel: float = 0.02,
                  refine_frac: float = 0.2) -> Onsets:
    """Impact clicks: high-pass, rectified envelope, threshold that follows the background, minimum gap.

    The high-pass removes hum and voices below the click band. Each onset is placed where the envelope first
    reaches refine_frac of that click's own peak, so loud and quiet clicks of the same shape are timed alike:
    a fixed absolute threshold would fire earlier on loud clicks and bias every flight time.
    """
    x = np.asarray(x, dtype=float)
    if x.size < int(0.05 * sr):
        return Onsets(np.empty(0), np.empty(0), 0.0)
    sos = butter(4, highpass_hz, "highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, x - x.mean())
    win = max(1, round(smooth_s * sr))
    env = uniform_filter1d(np.abs(y), win)
    peak = float(env.max())
    sigma_t = max(smooth_s / 2, 2.0 / sr)
    if peak <= 0:
        return Onsets(np.empty(0), np.empty(0), sigma_t)

    # Background from a running median on a 1 ms grid (a full-rate median over half a second would be slow);
    # a median ignores the short clicks themselves, so the threshold rides on hiss and hum only.
    hop = max(1, int(sr * 0.001))
    coarse = env[: env.size // hop * hop].reshape(-1, hop).mean(axis=1)
    if coarse.size == 0:
        return Onsets(np.empty(0), np.empty(0), sigma_t)
    bg = median_filter(coarse, size=max(3, int(background_s * 1000)), mode="nearest")
    thr = np.maximum(k * np.repeat(bg, hop), floor_rel * peak)
    thr = np.concatenate([thr, np.full(env.size - thr.size, thr[-1])])

    above = env > thr
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    gap = int(min_gap_s * sr)
    look = int(0.005 * sr)
    times, strength = [], []
    last = -gap
    for i in rising:
        if i - last < gap:
            continue
        last = i
        j = i + int(np.argmax(env[i:i + look]))
        level = refine_frac * env[j]
        k0 = j
        start = max(0, i - look)
        while k0 > start and env[k0 - 1] >= level:
            k0 -= 1
        # Linear interpolation between the two samples straddling the level: sub-sample timing.
        if k0 > 0 and env[k0] != env[k0 - 1]:
            frac = (level - env[k0 - 1]) / (env[k0] - env[k0 - 1])
            t = (k0 - 1 + float(np.clip(frac, 0.0, 1.0))) / sr
        else:
            t = k0 / sr
        times.append(t)
        strength.append(float(env[j]))
    return Onsets(np.array(times), np.array(strength), sigma_t)


@dataclass(frozen=True)
class Restitution:
    impact_times: np.ndarray  # the impact whose restitution each value describes (the onset between two flights)
    e: np.ndarray
    e_std: np.ndarray
    e_mean: float
    e_std_mean: float
    flight_times: np.ndarray


def _ratio_std(t1: float, t2: float, sigma_t: float) -> float:
    """Std of (c - b) / (b - a) when each onset a, b, c has timing error sigma_t; b is shared, so terms correlate."""
    da = t2 / t1**2
    db = 1 / t1 + t2 / t1**2
    dc = 1 / t1
    return float(sigma_t * np.sqrt(da**2 + db**2 + dc**2))


def weighted_mean(values: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    """Inverse-variance mean; its error is inflated by the Birge ratio when the values scatter more than their errors
    (restitution really does change with impact speed, so the per-bounce errors alone would be too optimistic)."""
    w = 1 / np.maximum(stds, 1e-12) ** 2
    m = float(np.sum(w * values) / np.sum(w))
    s = float(np.sqrt(1 / np.sum(w)))
    if len(values) > 1:
        chi2 = float(np.sum(w * (values - m) ** 2)) / (len(values) - 1)
        s *= max(1.0, np.sqrt(chi2))
    return m, s


def restitution_from_onsets(times: np.ndarray, sigma_t: float, min_flight_s: float = 0.02) -> Restitution:
    """e at each impact from consecutive flight times of a vertical bounce: e = T(n+1) / T(n).

    A flight under constant gravity lasts T = 2 v / g, and the take-off speed after an impact is e times the
    one before, so the ratio of flight times is e with g and every length cancelling: no scale, no gravity value,
    only timing. The longest run of onsets whose flights shrink like a bounce train is used; other sounds before
    or after it (a hand, a voice) are left out rather than turned into a restitution.
    """
    times = np.sort(np.asarray(times, dtype=float))
    T = np.diff(times)
    ok = T >= min_flight_s
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(len(T)):
        if not ok[i]:
            runs.append((start, i))
            start = i + 1
        elif i > start and not 0.1 < T[i] / T[i - 1] < 1.0 + 3 * _ratio_std(T[i - 1], T[i], sigma_t):
            runs.append((start, i))
            start = i  # the flight that broke the run can still begin the next one
    runs.append((start, len(T)))
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    # A sound just before the drop (a clap, the hand letting go) can pass the "shrinking flight" test once; its
    # ratio then stands out from the train's own, which changes only slowly from bounce to bounce.
    while hi - lo >= 4:
        e_run = T[lo + 1:hi] / T[lo:hi - 1]
        med = float(np.median(e_run))
        if abs(e_run[0] - med) > MAX_E_JUMP:
            lo += 1
        elif abs(e_run[-1] - med) > MAX_E_JUMP:
            hi -= 1
        else:
            break
    flights = T[lo:hi]
    if len(flights) < 2:
        raise AcousticError("Fewer than three impacts in a row that sound like a bouncing object.")
    e = flights[1:] / flights[:-1]
    e_std = np.array([_ratio_std(a, b, sigma_t) for a, b in pairwise(flights)])
    # Flight n spans onsets n and n+1, so the ratio of flights n and n+1 belongs to the impact at onset n+1.
    impact_times = times[lo + 1:lo + len(flights)]
    m, s = weighted_mean(e, e_std)
    return Restitution(impact_times, e, e_std, m, s, flights)


@dataclass(frozen=True)
class Tone:
    freq_hz: float
    halfwidth_hz: float  # stated interval: freq_hz +- halfwidth_hz, one frequency bin of the clip's own length
    snr_db: float


def _peak_offset(a: float, b: float, c: float) -> float:
    """Vertex of the parabola through three log-magnitude bins, in bins from the middle one."""
    den = a - 2 * b + c
    return 0.0 if den == 0 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def dominant_frequency(x: np.ndarray, sr: int = SAMPLE_RATE, fmin: float = 20.0, fmax: float | None = None,
                       min_snr_db: float = 15.0) -> Tone:
    """The strongest steady tone: Hann window, zero-padded FFT, parabolic peak interpolation.

    The interval is +-1/duration, the clip's natural frequency resolution. Interpolation usually lands far closer
    than that, but the stated interval is the one that holds without assuming the tone is perfectly steady.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 64:
        raise AcousticError("The soundtrack is too short to measure a frequency.")
    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= fmin) & (freqs <= (fmax or sr / 2))
    idx = np.flatnonzero(band)
    if idx.size < 3:
        raise AcousticError("The frequency band is empty.")
    i = int(idx[np.argmax(spec[idx])])
    floor = float(np.median(spec[idx])) + 1e-30
    snr_db = float(20 * np.log10(spec[i] / floor))
    if snr_db < min_snr_db:
        raise AcousticError(f"No clear tone (strongest peak only {snr_db:.0f} dB above the background).")
    lg = np.log(spec[max(i - 1, 0):i + 2] + 1e-30)
    off = _peak_offset(*lg) if lg.size == 3 else 0.0
    return Tone(float((i + off) * sr / nfft), float(sr / n), snr_db)


@dataclass(frozen=True)
class Ridge:
    t: np.ndarray  # seconds, frame centres
    f: np.ndarray  # Hz
    resolution_hz: float  # one STFT bin before interpolation


def frequency_ridge(x: np.ndarray, sr: int = SAMPLE_RATE, fmin: float = 20.0, fmax: float = 5000.0,
                    window_s: float = 0.1, hop_s: float = 0.02, max_step: float = 0.08,
                    min_snr_db: float = 12.0) -> Ridge:
    """Follow one tone through a spectrogram, starting from its loudest moment and walking both ways in time.

    Each step searches only near the previous frequency (max_step relative), so a louder unrelated sound
    elsewhere in the spectrum can't hijack the track; the ridge ends where the tone sinks into the background.
    """
    nper = int(window_s * sr)
    noverlap = nper - int(hop_s * sr)
    f, t, S = spectrogram(np.asarray(x, dtype=float), fs=sr, window="hann", nperseg=nper, noverlap=noverlap,
                          nfft=4 * nper, mode="magnitude", detrend="constant")
    band = (f >= fmin) & (f <= fmax)
    f, S = f[band], S[band]
    if S.shape[1] < 3 or f.size < 3:
        raise AcousticError("The soundtrack is too short to follow a tone.")
    logS = np.log(S + 1e-30)
    floor = np.median(S, axis=0) + 1e-30
    snr = 20 * np.log10(S.max(axis=0) / floor)
    anchor = int(np.argmax(snr))
    if snr[anchor] < min_snr_db:
        raise AcousticError("No tone stands out of the background anywhere in the soundtrack.")
    df = f[1] - f[0]

    def peak_near(col: int, f_prev: float) -> float | None:
        lo = np.searchsorted(f, f_prev * (1 - max_step) - df)
        hi = np.searchsorted(f, f_prev * (1 + max_step) + df) + 1
        lo, hi = max(lo, 1), min(hi, f.size - 1)
        if hi <= lo:
            return None
        i = lo + int(np.argmax(S[lo:hi, col]))
        if 20 * np.log10(S[i, col] / floor[col]) < min_snr_db:
            return None
        return float(f[i] + df * _peak_offset(logS[i - 1, col], logS[i, col], logS[i + 1, col]))

    i0 = int(np.argmax(S[:, anchor]))
    i0 = int(np.clip(i0, 1, f.size - 2))
    f0 = float(f[i0] + df * _peak_offset(logS[i0 - 1, anchor], logS[i0, anchor], logS[i0 + 1, anchor]))
    track = {anchor: f0}
    for step in (1, -1):
        col, cur = anchor + step, f0
        while 0 <= col < S.shape[1]:
            nxt = peak_near(col, cur)
            if nxt is None:
                break
            track[col] = cur = nxt
            col += step
    cols = np.array(sorted(track))
    return Ridge(t[cols], np.array([track[c] for c in cols]), float(df))


def _ridge_noise(f: np.ndarray) -> float:
    """Per-frame frequency noise of a smooth ridge, from second differences (insensitive to the slow trend)."""
    if f.size < 4:
        return 0.0
    return float(np.std(np.diff(f, 2)) / np.sqrt(6))


def spin_down_from_sound(x: np.ndarray, sr: int = SAMPLE_RATE, blade_count: int | None = None, fmin: float = 20.0,
                         fmax: float = 5000.0, min_duration_s: float = 1.0, min_drop: float = 0.05) -> dict[str, Any]:
    """Friction law of a spin-down heard as a falling tone (a fan's blade-pass whine, a spinning top's hum).

    The ridge frequency is proportional to the rotation rate, so integrating it gives a phase that turns exactly
    like the spinning part, up to a constant factor. Every spin-down law in models.py keeps its form under a
    constant rescaling of the rate (only c and q rescale), so the law is identified without knowing that factor.
    The factor is the blade count: given it, the coefficients are converted to the rotor's own rates.

    For the fit the rate is normalized to start at 2 pi rad/s (one turn per second), where the models' parameter
    bounds and the smoothing of sgn() near zero speed were designed to work, then converted back.
    """
    ridge = frequency_ridge(x, sr, fmin, fmax)
    t, f = ridge.t, ridge.f
    if t[-1] - t[0] < min_duration_s or f.size < 20:
        raise AcousticError(f"The tone could be followed for only {t[-1] - t[0]:.1f} s; at least {min_duration_s:.0f} s is needed.")
    if (f[0] - f[-1]) / f[0] < min_drop:
        raise AcousticError(f"The tone stays near {np.median(f):.1f} Hz: nothing is spinning down.")
    f_ref = float(f[0])
    w_n = 2 * np.pi * f / f_ref
    phase = cumulative_trapezoid(w_n, t, initial=0.0)
    c = Coordinates("rotation", t - t[0], phase[:, None], ("phi",), {})
    fits = [fit_model(MODELS[m], c) for m in SPIN_LAWS]
    verdict = assess(c, fits)
    out: dict[str, Any] = {
        "t_start": float(t[0]), "t_end": float(t[-1]), "f_start_hz": float(f[0]), "f_end_hz": float(f[-1]),
        "ridge": [[round(float(a), 4), round(float(b), 3)] for a, b in zip(t, f, strict=True)],
        "status": verdict.status, "reason": verdict.reason, "blade_count": blade_count,
        "ranking": [{"model_id": r.model_id, "title": MODELS[r.model_id].title, "bic": r.bic,
                     "heldout_nrmse": r.heldout_nrmse} for r in verdict.ranked],
        "law": None, "acoustic_params": [], "rotor_params": [],
    }
    if verdict.winner is None:
        return out
    spec = MODELS[verdict.winner.model_id]
    est = bootstrap(spec, c, verdict.winner, reps=24)
    out["law"] = {"model_id": spec.id, "title": spec.title, "latex": spec.latex}
    # Back from normalized phase to the tone itself: f = f_ref * w_n / (2 pi).
    acoustic = {"c": (f_ref / (2 * np.pi), "c_f", "Hz/s", "tone falls this many Hz per second"),
                "k": (1.0, "k", "1/s", "relative decay rate, the same for tone and rotor"),
                "q": (2 * np.pi / f_ref, "q_f", "", "per cycle of the tone")}
    for name in spec.param_names:
        scale, label, unit, meaning = acoustic[name]
        out["acoustic_params"].append({"name": label, "value": est[name].value * scale, "std": est[name].std * scale,
                                       "unit": unit, "meaning": meaning})
    if blade_count:
        s = f_ref / blade_count  # the rotor's starting rate in turns per second
        rotor = {"c": s, "k": 1.0, "q": 1 / s}
        for p in spec.params:
            v, sd = est[p.name].value * rotor[p.name], est[p.name].std * rotor[p.name]
            out["rotor_params"].append({"name": p.name, "latex": p.latex, "value": v, "std": sd, "unit": p.unit})
        out["rotor_start_rate_hz"] = s
    return out


def mean_rate(ridge_t: np.ndarray, ridge_f: np.ndarray, a: float, b: float, resolution_hz: float = 0.0) -> tuple[float, float]:
    """Mean tone frequency over [a, b] with its error, from the per-frame scatter (and a resolution floor)."""
    sel = (ridge_t >= a) & (ridge_t <= b)
    if sel.sum() < 4:
        raise AcousticError("The tone and the camera overlap for too short a time to compare.")
    f = ridge_f[sel]
    std = max(_ridge_noise(f) / np.sqrt(f.size), 0.01 * resolution_hz)
    return float(f.mean()), float(std)


def fuse(camera: tuple[float, float], microphone: tuple[float, float], max_sigma: float = AGREE_SIGMA) -> dict[str, Any]:
    """Combine two measurements of one quantity by inverse variance, but only if they agree within max_sigma.

    When they disagree the two are reported side by side and flagged: averaging would hide that one sensor is
    measuring something else (a different sound, a missed impact), which is exactly what the user should see.
    """
    (vc, sc), (vm, sm) = camera, microphone
    sigma = float(np.hypot(sc, sm))
    z = abs(vc - vm) / sigma if sigma > 0 else (0.0 if vc == vm else np.inf)
    out: dict[str, Any] = {"camera": {"value": vc, "std": sc}, "microphone": {"value": vm, "std": sm},
                           "z": float(z), "agree": bool(z <= max_sigma), "fused": None}
    if out["agree"] and sc > 0 and sm > 0:
        w1, w2 = 1 / sc**2, 1 / sm**2
        out["fused"] = {"value": float((w1 * vc + w2 * vm) / (w1 + w2)), "std": float(np.sqrt(1 / (w1 + w2)))}
    return out
