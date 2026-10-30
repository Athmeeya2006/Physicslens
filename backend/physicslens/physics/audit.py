"""Physics audit: does the tracked motion obey physics at all? A lie detector for generated video and world models.

The rest of the pipeline asks "which law explains this motion?". The audit asks a sharper question: "does this
motion break a law that every real object obeys?" Each check measures one kind of violation, with its size and how
many standard errors it lies from what physics allows:

- gravity: g against 9.81 m/s² when the clip has a real scale; with no scale, every flight must share one g and
  one direction of gravity
- energy: a bounce that returns more speed than it hit with, a free flight that gains energy, an oscillator whose
  swing grows
- period drift: an oscillator whose frequency wanders with no change of setup to explain it
- law: no model, including the agent's own, forecasts the held-out motion of a named passive system
- kinematics: an object that jumps further in one frame than its speed allows (teleporting), or whose mask area
  steps abruptly (morphing)

The verdict is "violates_physics", "consistent" or "cant_judge". False accusations are worse than misses, so
"cant_judge" is the default whenever a precondition fails (too short, poorly tracked, variable frame rate, moving
camera), and a violation needs BOTH a large z-score AND a practically large magnitude: a 5 sigma deviation can
still be a real-world effect the noise model doesn't know about (a reference sheet not quite in the plane of
motion, air drag on a light object), and a large deviation measured on noise is no evidence at all.

Pure functions: no network; save_input and load_input are the only file access (replaying a changed rule on stored runs).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ellipk

from ..sigproc.cleanup import detect_impacts
from ..sigproc.trajectory import Trajectory
from .collisions import detect_bounces
from .coords import length_is_reliable, to_coordinates
from .fit import dominant_omega

EARTH_G = 9.81

# Significance: 5 sigma, the particle-physics discovery standard. Tracking noise is not perfectly Gaussian
# (mask jitter, motion blur), so the tails are heavier than a normal distribution's; 5 sigma keeps chance
# alarms rare even after the dozens of windows and impacts one clip can be tested on.
Z_VIOLATION = 5.0
# Frame-to-frame jumps are tested at every sample of the clip, against a robust (median-based) noise scale whose
# own error is larger, and SAM's mask jitter has heavy tails: hence a stricter bar.
Z_KINEMATIC = 8.0

# Practical magnitudes, each larger than the real-world effects the uncertainty doesn't capture.
# A reference object 10 to 20 cm in front of or behind the plane of motion (common when filming by hand) scales
# every length, and so g, by 10 to 20%; 25% stays clear of that while still catching a world with g = 14.
G_REL_TOL = 0.25
# Flights of one clip share one g; drag on a fast throw versus a slow hop shifts the apparent curvature by a few
# percent, and short flights measure it poorly.
FLIGHT_G_REL_TOL = 0.25
FLIGHT_ANGLE_TOL_DEG = 10.0  # camera lens distortion bends a parabola near the frame edge by a few degrees at most
# Real balls keep at most about 0.9 of their speed; 1.1 leaves room for an impact time misplaced by a frame.
MAX_RESTITUTION = 1.1
FLIGHT_ENERGY_GAIN = 0.15  # fraction of the flight's energy scale gained with nothing pushing
AMPLITUDE_GROWTH = 0.2  # a passive oscillator's swing growing by more than this over the clip
PERIOD_DRIFT = 0.04  # relative change of frequency over the clip; pendulum anharmonicity is corrected first
NO_LAW_NRMSE = 0.5  # the best law misses the held-out motion by half its spread (verification rejects at 0.15 to 0.3)
JUMP_SPAN_FRACTION = 0.05  # a teleport must move at least this share of the motion's extent ...
JUMP_OBJECT_FRACTION = 0.5  # ... or half the object's own size, when that is known
MORPH_FACTOR = 1.5  # mask area stepping up or down by 50% between neighbouring stretches of frames

# Preconditions: below these the audit says "can't judge" rather than risk an accusation.
MIN_DURATION_S = 1.5
MIN_SAMPLES = 30
MIN_COVERAGE = 0.7  # share of the clip's frames in which the object was tracked
MIN_CONFIDENCE = 0.5  # SAM's median confidence on the object
MIN_MOTION_SNR = 20.0  # extent of the motion over the tracking noise
CAMERA_DRIFT_FRACTION = 0.15  # a still reference may wander this share of its own size (mask jitter) ...
CAMERA_DRIFT_MIN_PX = 6.0  # ... and at least this many pixels

MIN_FLIGHT_SAMPLES = 10
MIN_FLIGHT_G_SNR = 20.0
EDGE_TRIM = 1  # samples dropped each side of an impact, as in the bounce model
MIN_CYCLES_PER_WINDOW = 1.5
MIN_WINDOWS = 3
MORPH_WINDOW = 5
# Contact: the velocity just after an event differs from just before by more than this share of the faster one.
# A teleport keeps its velocity (only the position breaks); a tracker hopping to a lookalike at a collision (a
# Newton's cradle) or a mask swelling with sand spray on impact happens at a velocity break, a contact. Real
# footage in the control set raised both, so a jump or morph at a contact is not counted against the video.
CONTACT_VELOCITY_CHANGE = 0.6
CONTACT_WINDOW = 4  # frames on each side used for the velocities
BOOTSTRAP_REPS = 100

Verdict = Literal["consistent", "violates_physics", "cant_judge"]
CheckStatus = Literal["pass", "violation", "skipped"]
REAL_SCALE_EXCLUDED = ("none", "gravity")
PASSIVE_FAMILIES = ("pendulum", "oscillator", "projectile")
DYNAMIC_CHECKS = ("gravity", "energy", "period", "law")
SCALE_BASIS = {"user": "the pendulum length you gave", "object_size": "the object size you gave",
               "a4_sheet": "the A4 sheet in view", "us_letter_sheet": "the letter sheet in view"}


@dataclass(frozen=True)
class LawEvidence:
    """What the verification step found, summarized for the audit."""

    status: str  # accepted | ambiguous | no_fit
    model_id: str | None = None
    heldout_nrmse: float | None = None  # best candidate's held-out forecast error over the signal's spread
    heldout_vs_noise: float | None = None  # the same error over the tracking noise
    g: float | None = None  # from the verified law, in m/s², when it has one
    g_std: float | None = None


@dataclass(frozen=True)
class AuditInput:
    traj: Trajectory  # the whole track in meters (on the nominal scale when nothing gives one)
    family: str
    calibration_source: str
    px_per_m: float
    scale_rel_std: float = 0.0
    law: LawEvidence | None = None
    object_size_px: float | None = None
    areas_px: np.ndarray | None = None  # mask area per sample, aligned with traj
    confidence: np.ndarray | None = None
    coverage: float = 1.0
    frame_rate_varies: bool | None = None  # None: not checked
    reference_px: np.ndarray | None = None  # (N, 2) centers of the scale reference, which should not move
    reference_size_px: float | None = None
    regime_change_s: float | None = None
    lookalikes: bool = False  # the scene shows objects like the tracked one (a Newton's cradle): identity can't be followed


@dataclass
class Check:
    check: str  # gravity | energy | period | law | kinematics
    title: str
    status: CheckStatus
    text: str
    value: float | None = None
    std: float | None = None
    expected: float | None = None
    unit: str = ""
    z: float | None = None


@dataclass
class Flight:
    start: int
    stop: int
    lo: int  # impact index (or clip edge) before and after the flight
    hi: int
    t0: float
    t_start: float
    t_end: float
    cx: np.ndarray  # quadratic coefficients in (t - t0): [c0, c1, c2]
    cy: np.ndarray
    cov: np.ndarray = field(repr=False)  # (XᵀX)⁻¹ for the shared design
    sx: float = 0.0  # residual scale per axis
    sy: float = 0.0

    @property
    def accel(self) -> np.ndarray:
        return np.array([2 * self.cx[2], 2 * self.cy[2]])

    @property
    def accel_cov(self) -> np.ndarray:
        return np.diag([4 * self.sx**2 * self.cov[2, 2], 4 * self.sy**2 * self.cov[2, 2]])

    def vy(self, t: float) -> tuple[float, float]:
        """Vertical velocity at time t and its standard error."""
        grad = np.array([0.0, 1.0, 2 * (t - self.t0)])
        return float(grad @ self.cy), float(self.sy * np.sqrt(grad @ self.cov @ grad))

    def y(self, t: float) -> float:
        tau = t - self.t0
        return float(self.cy[0] + self.cy[1] * tau + self.cy[2] * tau * tau)


# ---------- small helpers ----------

def noise_sigma(xy: np.ndarray) -> float:
    """Position noise from second differences (smooth motion barely contributes at video rates); robust to impacts."""
    d2 = np.diff(xy, 2, axis=0)
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0)
    return float(np.max(mad) * 1.4826 / np.sqrt(6))


def _robust_scale(v: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def pm(value: float, std: float | None, unit: str = "") -> str:
    """'14.2 ± 0.6 m/s²': rounded to the uncertainty's first significant figure (two when that digit is 1 or 2)."""
    unit = f" {unit}" if unit else ""
    if std is None or not (std > 0) or not math.isfinite(std):
        return f"{value:.3g}{unit}"
    exp = math.floor(math.log10(std))
    d = min(6, max(0, -exp + (1 if std / 10**exp < 3 else 0)))
    return f"{value:.{d}f} ± {std:.{d}f}{unit}"


def _trend(t: np.ndarray, y: np.ndarray, std: np.ndarray) -> tuple[float, float]:
    """Weighted straight-line slope and its standard error, inflated when the points scatter more than claimed."""
    w = 1.0 / np.maximum(std, 1e-12) ** 2
    tm = np.sum(w * t) / np.sum(w)
    sxx = float(np.sum(w * (t - tm) ** 2))
    slope = float(np.sum(w * (t - tm) * y) / sxx)
    intercept = float(np.sum(w * y) / np.sum(w))
    resid = y - intercept - slope * (t - tm)
    dof = len(t) - 2
    chi2 = float(np.sum(w * resid**2) / dof) if dof > 0 else 1.0
    return slope, float(np.sqrt(max(chi2, 1.0) / sxx))


def pretty(unit: str) -> str:
    return unit.replace("^2", "²")


def _units(a: AuditInput) -> tuple[bool, float, str]:
    """Whether lengths are real, the factor from track units to displayed units, and the length unit."""
    real = a.calibration_source not in REAL_SCALE_EXCLUDED
    return real, (1.0 if real else a.px_per_m), ("m" if real else "px")


# ---------- preconditions ----------

def preconditions(a: AuditInput, sigma: float) -> list[str]:
    """Reasons the clip can't be judged at all; empty when it can."""
    t, xy = a.traj.t, a.traj.xy
    reasons: list[str] = []
    duration = float(t[-1] - t[0]) if len(t) else 0.0
    if len(t) < MIN_SAMPLES or duration < MIN_DURATION_S:
        reasons.append(f"The tracked motion lasts {duration:.1f} s ({len(t)} samples); at least {MIN_DURATION_S:.1f} s "
                       f"and {MIN_SAMPLES} samples are needed to judge it.")
        return reasons
    if a.coverage < MIN_COVERAGE:
        reasons.append(f"The object was tracked in only {a.coverage:.0%} of the frames, so gaps could hide or fake a violation.")
    if a.confidence is not None and len(a.confidence) and np.any(a.confidence > 0):
        conf = float(np.median(a.confidence))
        if conf < MIN_CONFIDENCE:
            reasons.append(f"The tracker was unsure of the object (median confidence {conf:.2f}).")
    if a.frame_rate_varies:
        reasons.append("The video has a variable frame rate: frame times are uncertain, and timing is what physics is judged on.")
    if a.reference_px is not None and len(a.reference_px) >= 10:
        ref = np.asarray(a.reference_px, dtype=float)
        travel = float(np.max(np.percentile(ref, 95, axis=0) - np.percentile(ref, 5, axis=0)))
        allowed = max(CAMERA_DRIFT_FRACTION * (a.reference_size_px or 0.0), CAMERA_DRIFT_MIN_PX)
        if travel > allowed:
            reasons.append(f"The camera seems to move: the reference object, which should stand still, wanders {travel:.0f} px.")
    span = float(np.ptp(xy, axis=0).max())
    if span < MIN_MOTION_SNR * max(sigma, 1e-12):
        reasons.append(f"The motion is only {span / max(sigma, 1e-12):.0f} times the tracking noise; at least "
                       f"{MIN_MOTION_SNR:.0f} is needed.")
    return reasons


# ---------- free flights (projectile family) ----------

def find_flights(traj: Trajectory, sigma: float) -> list[Flight]:
    """Stretches between impacts that are clean free-fall parabolas."""
    n = len(traj)
    impacts = np.union1d(detect_impacts(traj), detect_bounces(traj)).astype(int)
    bounds = [0, *[int(i) for i in impacts], n]
    out: list[Flight] = []
    for lo, hi in pairwise(bounds):
        start = lo + (EDGE_TRIM if lo > 0 else 0)
        stop = hi - (EDGE_TRIM if hi < n else 0)
        if stop - start < MIN_FLIGHT_SAMPLES:
            continue
        f = _fit_flight(traj, start, stop, lo, hi, sigma)
        if f is not None:
            out.append(f)
    return out


def _fit_flight(traj: Trajectory, start: int, stop: int, lo: int, hi: int, sigma: float) -> Flight | None:
    t, xy = traj.t[start:stop], traj.xy[start:stop]
    t0 = float(t.mean())
    X = np.column_stack([np.ones_like(t), t - t0, (t - t0) ** 2])
    coef, *_ = np.linalg.lstsq(X, xy, rcond=None)
    resid = xy - X @ coef
    dof = max(len(t) - 3, 1)
    s = np.sqrt(np.sum(resid**2, axis=0) / dof)
    # A flight is free fall only if a parabola explains it to the tracking noise; a light object under heavy drag
    # (a paper bag) fails this and is left out, rather than being judged against a law it isn't following.
    floor = max(sigma, 1e-6)
    if np.max(s) > 3 * floor or np.ptp(xy[:, 1]) < 10 * floor:
        return None
    f = Flight(start, stop, lo, hi, t0, float(t[0]), float(t[-1]), coef[:, 0], coef[:, 1], np.linalg.inv(X.T @ X),
                  float(max(s[0], floor)), float(max(s[1], floor)))
    # Tiny hops merged by the impact detector average out to a nearly flat curve; only a flight whose own
    # curvature is measured to a few percent can speak for gravity.
    g, g_std = _g_of(f)
    return f if g >= MIN_FLIGHT_G_SNR * g_std else None


def _g_of(f: Flight) -> tuple[float, float]:
    a = f.accel
    mag = float(np.hypot(*a))
    u = a / max(mag, 1e-12)
    return mag, float(np.sqrt(u @ f.accel_cov @ u))


def _angle_of(f: Flight) -> tuple[float, float]:
    """Direction of the acceleration from straight down, degrees, and its standard error."""
    a = f.accel
    mag = float(np.hypot(*a))
    n = np.array([-a[1], a[0]]) / max(mag, 1e-12)
    return float(np.degrees(np.arctan2(a[0], -a[1]))), float(np.degrees(np.sqrt(n @ f.accel_cov @ n) / max(mag, 1e-12)))


def check_gravity_scale(a: AuditInput, flights: list[Flight]) -> Check:
    title = "Gravity against Earth's"
    if a.calibration_source in REAL_SCALE_EXCLUDED:
        return Check("gravity", title, "skipped", "No real scale in the clip, so g can't be compared with 9.81 m/s²; "
                                                    "flights are checked against each other instead.")
    g = std = None
    how = ""
    law = a.law
    if law is not None and law.status == "accepted" and law.g is not None and a.family in ("pendulum", "projectile"):
        g, std, how = law.g, law.g_std, "from the verified law"
    elif flights:
        gs = np.array([_g_of(f) for f in flights])
        w = 1 / gs[:, 1] ** 2
        g = float(np.sum(w * gs[:, 0]) / np.sum(w))
        std = float(np.sqrt(1 / np.sum(w)))
        std = float(np.hypot(std, g * a.scale_rel_std))
        how = f"from {len(flights)} free-fall flight{'s' if len(flights) > 1 else ''}"
    if g is None:
        return Check("gravity", title, "skipped", "No verified law gives g and no clean free-fall flight was found.")
    std = std if std and std > 0 else None
    z = abs(g - EARTH_G) / std if std else None
    rel = abs(g - EARTH_G) / EARTH_G
    basis = SCALE_BASIS.get(a.calibration_source, f"the {a.calibration_source.replace('_', ' ')} in view")
    # A wrong size typed in by the user moves g just as much as a fake world does, so the basis is always named.
    text = f"g = {pm(g, std, 'm/s²')} {how} (scale from {basis}), against 9.81 m/s²"
    if rel > G_REL_TOL and z is not None and z >= Z_VIOLATION:
        return Check("gravity", title, "violation", f"{text}: {rel:.0%} off, {z:.0f} standard errors.", g, std, EARTH_G, "m/s^2", z)
    return Check("gravity", title, "pass", f"{text}: within what the scale and noise allow.", g, std, EARTH_G, "m/s^2", z)


def check_gravity_consistency(a: AuditInput, flights: list[Flight]) -> list[Check]:
    if len(flights) < 2:
        return []
    _, k, unit = _units(a)
    gs = np.array([_g_of(f) for f in flights])
    w = 1 / gs[:, 1] ** 2
    mean = float(np.sum(w * gs[:, 0]) / np.sum(w))
    # Each flight against the weighted mean of all the others: a single odd flight is what this is looking for.
    worst: tuple[float, float, int] = (0.0, 0.0, -1)
    for i in range(len(flights)):
        others = np.arange(len(flights)) != i
        m = float(np.sum(w[others] * gs[others, 0]) / np.sum(w[others]))
        s = float(np.hypot(gs[i, 1], np.sqrt(1 / np.sum(w[others]))))
        z = abs(gs[i, 0] - m) / s
        if z > worst[0]:
            worst = (z, (gs[i, 0] - m) / m, i)
    z, rel, i = worst
    title = "One gravity for every flight"
    out = []
    g_unit = f"{unit}/s^2"
    if i >= 0 and z >= Z_VIOLATION and abs(rel) > FLIGHT_G_REL_TOL:
        out.append(Check("gravity", title, "violation",
                         f"Flight {i + 1} falls with {pm(gs[i, 0] * k, gs[i, 1] * k, pretty(g_unit))}, {abs(rel):.0%} "
                         f"{'more' if rel > 0 else 'less'} than the other flights ({z:.0f} standard errors): gravity changed mid-clip.",
                         float(gs[i, 0] * k), float(gs[i, 1] * k), float(mean * k), g_unit, z))
    else:
        out.append(Check("gravity", title, "pass",
                         f"All {len(flights)} flights share one g ({pm(mean * k, np.sqrt(1 / np.sum(w)) * k, pretty(g_unit))}).",
                         float(mean * k), float(np.sqrt(1 / np.sum(w)) * k), None, g_unit, z))
    angles = np.array([_angle_of(f) for f in flights])
    wa = 1 / angles[:, 1] ** 2
    worst_a: tuple[float, float, int] = (0.0, 0.0, -1)
    for i in range(len(flights)):
        others = np.arange(len(flights)) != i
        m = float(np.sum(wa[others] * angles[others, 0]) / np.sum(wa[others]))
        s = float(np.hypot(angles[i, 1], np.sqrt(1 / np.sum(wa[others]))))
        z = abs(angles[i, 0] - m) / s
        if z > worst_a[0]:
            worst_a = (z, angles[i, 0] - m, i)
    z, diff, i = worst_a
    title = "One direction of gravity"
    if i >= 0 and z >= Z_VIOLATION and abs(diff) > FLIGHT_ANGLE_TOL_DEG:
        out.append(Check("gravity", title, "violation",
                         f"In flight {i + 1} gravity points {abs(diff):.0f}° away from the other flights ({z:.0f} standard errors).",
                         float(diff), float(angles[i, 1]), 0.0, "deg", z))
    else:
        out.append(Check("gravity", title, "pass", "Gravity points the same way in every flight.", None, None, None, "", z))
    return out


def _impact_time(f: Flight, g: Flight, dt: float) -> float:
    """Where the falling parabola meets the rising one: the moment of impact, between the two flights."""
    lo, hi = f.t_end - dt, g.t_start + dt

    def diff(t: float) -> float:
        return f.y(t) - g.y(t)

    if diff(lo) * diff(hi) < 0:
        return float(brentq(diff, lo, hi))
    grid = np.linspace(lo, hi, 64)
    return float(grid[np.argmin(np.abs([diff(x) for x in grid]))])


def check_restitution(a: AuditInput, flights: list[Flight]) -> Check | None:
    impacts = []
    for f, g in pairwise(flights):
        if f.hi != g.lo:  # not separated by exactly one impact
            continue
        ti = _impact_time(f, g, a.traj.dt)
        vb, sb = f.vy(ti)
        va, sa = g.vy(ti)
        # A floor bounce: falling before, rising after, both speeds clearly measured.
        if not (vb < -5 * sb and va > 5 * sa):
            continue
        e = -va / vb
        e_std = abs(e) * float(np.hypot(sa / va, sb / vb))
        impacts.append((ti, e, e_std))
    if not impacts:
        return None
    title = "Bounces lose energy"
    zs = [(e - 1) / s if s > 0 else 0.0 for _, e, s in impacts]
    i = int(np.argmax(zs))
    ti, e, s = impacts[i]
    gain = e * e - 1
    if e > MAX_RESTITUTION and zs[i] >= Z_VIOLATION:
        return Check("energy", title, "violation",
                     f"The bounce at {ti:.2f} s returns more speed than it hit with: e = {pm(e, s)} "
                     f"(at most 1 for a real bounce), so it gains {gain:.0%} energy ({zs[i]:.0f} standard errors).",
                     e, s, 1.0, "", zs[i])
    top = max(x[1] for x in impacts)
    return Check("energy", title, "pass",
                 f"Every one of {len(impacts)} bounces keeps at most e = {top:.2f} of its speed.", top, None, 1.0, "", zs[i])


def _flight_energy_gain(traj: Trajectory, f: Flight, rng: np.random.Generator) -> tuple[float, float] | None:
    """Energy gained over one flight relative to its energy scale, with a bootstrap standard error.

    The flight's own mean acceleration stands in for gravity, so a wrong g doesn't count twice; what is left is a
    force that changes along the flight. Drag takes energy out; only a push puts it in.
    """
    t, xy = traj.t[f.start:f.stop], traj.xy[f.start:f.stop]
    if len(t) < 16:
        return None
    tau = t - f.t0
    X = np.column_stack([np.ones_like(tau), tau, tau**2, tau**3])
    pinv = np.linalg.pinv(X)
    coef = pinv @ xy
    fitted = X @ coef
    resid = xy - fitted

    def gain(c: np.ndarray) -> float:
        G = 2 * c[2]
        v = np.column_stack([c[1, j] + 2 * c[2, j] * tau + 3 * c[3, j] * tau**2 for j in range(2)])
        r = X @ c
        energy = 0.5 * np.sum(v**2, axis=1) - r @ G
        g_mag = float(np.hypot(*G)) or 1.0
        height = r @ (-G / g_mag)
        scale = 0.5 * float(np.mean(np.sum(v**2, axis=1))) + g_mag * float(np.ptp(height))
        return float((energy[-1] - energy[0]) / max(scale, 1e-12))

    base = gain(coef)
    boots = [gain(pinv @ (fitted + resid[rng.integers(0, len(t), len(t))])) for _ in range(BOOTSTRAP_REPS)]
    return base, float(np.std(boots, ddof=1))


def check_flight_energy(a: AuditInput, flights: list[Flight]) -> Check | None:
    rng = np.random.default_rng(0)
    gains = [(i, *r) for i, f in enumerate(flights) if (r := _flight_energy_gain(a.traj, f, rng)) is not None]
    if not gains:
        return None
    title = "Free flight doesn't gain energy"
    zs = [g / s if s > 0 else 0.0 for _, g, s in gains]
    j = int(np.argmax(zs))
    i, g, s = gains[j]
    if g > FLIGHT_ENERGY_GAIN and zs[j] >= Z_VIOLATION:
        return Check("energy", title, "violation",
                     f"In flight {i + 1} the object gains {g:.0%} energy with nothing pushing it ({zs[j]:.0f} standard errors).",
                     g, s, 0.0, "", zs[j])
    return Check("energy", title, "pass", f"No flight gains energy beyond the noise (largest change {g:+.0%}).", g, s, 0.0, "", zs[j])


# ---------- oscillators ----------

@dataclass
class Windows:
    t: np.ndarray
    omega: np.ndarray
    omega_std: np.ndarray
    amp: np.ndarray
    amp_std: np.ndarray
    angular: bool  # amplitudes are swing angles (radians), corrected for the pendulum's anharmonicity


def oscillation_windows(traj: Trajectory, family: str) -> Windows | str:
    """Frequency and amplitude in consecutive non-overlapping windows, or why they can't be measured."""
    angular = False
    q: np.ndarray | None = None
    if family == "pendulum":
        try:
            c = to_coordinates(traj, "pendulum")
            if length_is_reliable(c):
                q, angular = c.q[:, 0], True
        except ValueError:
            q = None
    if q is None:
        # The principal axis of the path: exact for a spring, a fine stand-in for a small swing.
        xy = traj.xy - traj.xy.mean(axis=0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        q = xy @ vt[0]
    q = q - q.mean()
    t = traj.t
    omega0 = dominant_omega(t, q)
    if omega0 <= 0:
        return "no oscillation was found"
    period = 2 * np.pi / omega0
    span = float(t[-1] - t[0])
    n_win = int(span // (MIN_CYCLES_PER_WINDOW * period))
    if n_win < MIN_WINDOWS:
        return (f"the clip holds {span / period:.1f} cycles; at least {MIN_WINDOWS * MIN_CYCLES_PER_WINDOW:.1f} are needed "
                "to follow frequency and amplitude over time")
    noise = noise_sigma(np.column_stack([q, q]))
    edges = np.linspace(t[0], t[-1] + 1e-9, n_win + 1)
    rows = []
    for lo, hi in pairwise(edges):
        idx = np.nonzero((t >= lo) & (t < hi))[0]
        if len(idx) < 12:
            continue
        tw, qw = t[idx] - t[idx].mean(), q[idx]
        X = np.column_stack([np.ones_like(tw), np.cos(omega0 * tw), np.sin(omega0 * tw)])
        lin, *_ = np.linalg.lstsq(X, qw, rcond=None)

        def resid(p: np.ndarray, tw: np.ndarray = tw, qw: np.ndarray = qw) -> np.ndarray:
            return p[0] + p[1] * np.cos(p[3] * tw) + p[2] * np.sin(p[3] * tw) - qw

        sol = least_squares(resid, [*lin, omega0], x_scale="jac")
        _, A, B, w = sol.x
        amp = float(np.hypot(A, B))
        if amp < 10 * max(noise, 1e-12):
            continue  # a swing that has died down: its frequency is mostly noise
        s2 = max(float(np.sum(sol.fun**2)) / max(len(idx) - 4, 1), noise**2)
        try:
            cov = s2 * np.linalg.inv(sol.jac.T @ sol.jac)
        except np.linalg.LinAlgError:
            continue
        u = np.array([0.0, A, B, 0.0]) / max(amp, 1e-12)
        w_std = float(np.sqrt(max(cov[3, 3], 0.0)))
        a_std = float(np.sqrt(max(u @ cov @ u, 0.0)))
        if angular and amp < np.pi * 0.95:
            # A pendulum swinging through amplitude A runs slow by pi / (2 K(sin^2(A/2))): as damping shrinks the swing,
            # the frequency rises. Divided out, a healthy pendulum's frequency is the same in every window.
            factor = float(np.pi / (2 * ellipk(np.sin(amp / 2) ** 2)))
            w, w_std = w / factor, w_std / factor
        rows.append((float(t[idx].mean()), float(w), w_std, amp, a_std))
    if len(rows) < MIN_WINDOWS:
        return "the swing died down before enough cycles could be measured"
    arr = np.array(rows)
    return Windows(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], angular)


def check_oscillator(a: AuditInput) -> list[Check]:
    w = oscillation_windows(a.traj, a.family)
    if isinstance(w, str):
        return [Check("period", "Steady period", "skipped", f"Not checked: {w}."),
                Check("energy", "Swing doesn't grow", "skipped", f"Not checked: {w}.")]
    if a.regime_change_s is not None:
        why = f"Not checked: the dynamics change at {a.regime_change_s:.1f} s (a new setup), which explains a change of period or swing."
        return [Check("period", "Steady period", "skipped", why), Check("energy", "Swing doesn't grow", "skipped", why)]
    span = float(a.traj.t[-1] - a.traj.t[0])
    out = []
    slope, s = _trend(w.t, w.omega, w.omega_std)
    mean = float(np.average(w.omega, weights=1 / w.omega_std**2))
    drift = slope * span / mean
    z = abs(slope) / s
    first, last = w.omega[0] / (2 * np.pi), w.omega[-1] / (2 * np.pi)
    note = " after correcting for the swing's amplitude" if w.angular else ""
    if abs(drift) > PERIOD_DRIFT and z >= Z_VIOLATION:
        out.append(Check("period", "Steady period", "violation",
                         f"The frequency drifts by {drift:+.0%} over {span:.1f} s (from {first:.3f} to {last:.3f} Hz{note}) "
                         f"with no change of setup to explain it ({z:.0f} standard errors).", drift, s * span / mean, 0.0, "", z))
    else:
        out.append(Check("period", "Steady period", "pass",
                         f"The frequency holds at {mean / (2 * np.pi):.3f} Hz{note} (drift {drift:+.1%} over {span:.1f} s).",
                         drift, s * span / mean, 0.0, "", z))
    lam, s = _trend(w.t, np.log(w.amp), w.amp_std / w.amp)
    growth = math.expm1(lam * span)
    z = lam / s
    if growth > AMPLITUDE_GROWTH and z >= Z_VIOLATION:
        out.append(Check("energy", "Swing doesn't grow", "violation",
                         f"The swing grows by {growth:.0%} over {span:.1f} s with nothing visibly driving it ({z:.0f} standard "
                         "errors); a passive oscillator can only lose energy.", growth, None, 0.0, "", z))
    else:
        out.append(Check("energy", "Swing doesn't grow", "pass",
                         f"The swing {'shrinks' if growth < 0 else 'changes'} by {growth:+.0%} over {span:.1f} s, as a passive "
                         "oscillator may.", growth, None, 0.0, "", z))
    return out


# ---------- the law, and kinematics ----------

def check_law(a: AuditInput, family_seen: bool = True) -> Check | None:
    """family_seen: the motion itself shows the family (a clean free flight, a steady swing). The family is the agent's
    reading of the scene; when it is wrong (a ball that falls, hits a block and stops, read as an oscillator), no law
    of that family fits real footage either, and "no law fits" would accuse a real video."""
    law = a.law
    if law is None:
        return None
    title = "A law forecasts the motion"
    if law.status in ("accepted", "ambiguous"):
        what = "A verified law" if law.status == "accepted" else "More than one law"
        return Check("law", title, "pass", f"{what} forecasts the held-out motion.", law.heldout_nrmse, None, None, "", None)
    nrmse, ratio = law.heldout_nrmse, law.heldout_vs_noise
    if (a.family in PASSIVE_FAMILIES and family_seen and nrmse is not None and ratio is not None
            and nrmse > NO_LAW_NRMSE and ratio >= Z_VIOLATION):
        return Check("law", title, "violation",
                     f"It moves like a {a.family}, yet no law (the agent's own included) forecasts its held-out motion: the "
                     f"best misses by {nrmse:.0%} of the motion's spread, {ratio:.0f} times the tracking noise.",
                     nrmse, None, 0.0, "", ratio)
    if a.family in PASSIVE_FAMILIES and not family_seen:
        return Check("law", title, "skipped",
                     f"No law forecasts the motion, but the motion itself doesn't show {_FAMILY_SIGN[a.family]}, so the "
                     f"{a.family} reading may be wrong: not evidence of a violation.", nrmse, None, None, "", ratio)
    return Check("law", title, "skipped",
                 "No law forecasts the motion, but that alone isn't a violation: pushed, driven or deformable motion has "
                 "no single law either.", nrmse, None, None, "", ratio)


_FAMILY_SIGN = {"projectile": "a clean free-fall flight", "pendulum": "a steady back-and-forth swing",
                "oscillator": "a steady back-and-forth swing"}


def jump_sizes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each step: how clearly it is a position break (the pair's size above its quiet neighbours), and its size.

    Read from second differences: a teleport between samples i and i + 1 leaves a pair +J, -J at i and i + 1 with
    quiet samples either side. An impact changes velocity, not position, so it leaves a single same-signed bump;
    a one- or two-frame tracker glitch that comes straight back leaves +J, -2J, +J or +J, -J, -J, +J, whose outer
    terms are as loud as the pair. Only the clean pair counts. (A teleport across frames the tracker missed is
    spread over the interpolated gap and is not seen: a miss, never a false accusation.)
    """
    n = len(xy)
    out, size = np.zeros(max(n - 1, 0)), np.zeros(max(n - 1, 0))
    if n < 6:
        return out, size
    d2 = np.zeros_like(xy)
    d2[1:-1] = xy[2:] - 2 * xy[1:-1] + xy[:-2]
    mag = np.linalg.norm(d2, axis=1)
    for i in range(2, n - 3):
        a, b = d2[i], d2[i + 1]
        opposed = -float(a @ b) / max(mag[i] * mag[i + 1], 1e-300)
        if opposed <= 0:
            continue
        size[i] = min(mag[i], mag[i + 1]) * opposed
        out[i] = size[i] - max(mag[i - 1], mag[i + 2])
    return out, size


def at_contact(xy: np.ndarray, t: np.ndarray, before: int, after: int, sigma: float) -> bool:
    """Is there an impact between frame `before` and frame `after`: the object slows sharply or turns back?

    The motion just before is fitted with steady acceleration and carried forward. An impact is a velocity after
    the event far below that forecast, or pointing back against it (a ball landing in sand, a bob stopped by its
    neighbour). A swing turning round keeps its acceleration, so it is no impact; and speeding up from nowhere,
    as an object in a generated video may, is not excused either.
    """
    w = min(2 * CONTACT_WINDOW, before, len(xy) - 1 - after)
    if w < 3:
        return True  # too close to an end to fit either side: give the video the benefit of the doubt
    tb, ta = t[before - w:before + 1], t[after:after + w + 1]
    pre = [np.polyfit(tb - t[after], xy[before - w:before + 1, d], 2) for d in range(xy.shape[1])]
    post = [np.polyfit(ta - t[after], xy[after:after + w + 1, d], 2) for d in range(xy.shape[1])]
    v_pred = np.array([c[1] for c in pre])  # derivative at t[after] of the curve fitted before the event
    v_seen = np.array([c[1] for c in post])
    floor = 4 * sigma / (ta[-1] - ta[0])  # velocities below the noise mean nothing
    speed_pred, speed_seen = float(np.hypot(*v_pred)), float(np.hypot(*v_seen))
    if max(speed_pred, speed_seen) <= floor:
        return False
    if float(np.hypot(*(v_seen - v_pred))) <= CONTACT_VELOCITY_CHANGE * max(speed_pred, speed_seen, floor):
        return False  # the velocity carries on
    slowed = speed_seen < 0.5 * speed_pred
    turned = speed_pred > floor and float(np.dot(v_seen, v_pred)) < 0
    return slowed or turned


def check_jumps(a: AuditInput, sigma: float) -> Check:
    """Teleporting: a break in position between two frames, judged against the noise and the size of the motion."""
    _, k, unit = _units(a)
    gaps, sizes = jump_sizes(a.traj.xy)
    title = "No teleporting"
    if a.lookalikes:
        return Check("kinematics", title, "skipped", "Not checked: objects like this one are in view, and a tracker can "
                     "follow the wrong one (in a Newton's cradle the motion really does pass from ball to ball).")
    if not len(gaps):
        return Check("kinematics", title, "skipped", "Not checked: the track is too short.")
    scale = max(_robust_scale(gaps), 0.5 * sigma, 1e-12)
    z = (gaps - np.median(gaps)) / scale
    i = int(np.argmax(z))
    span = float(np.ptp(a.traj.xy, axis=0).max())
    practical = JUMP_SPAN_FRACTION * span
    if a.object_size_px:
        practical = max(practical, JUMP_OBJECT_FRACTION * a.object_size_px / a.px_per_m)
    jump = float(sizes[i])
    if z[i] >= Z_KINEMATIC and gaps[i] > practical and at_contact(a.traj.xy, a.traj.t, i, i + 1, sigma):
        return Check("kinematics", title, "skipped",
                     f"At {a.traj.t[i + 1]:.2f} s the track jumps {jump * k:.3g} {unit}, but at a contact (the velocity changes "
                     "there too): a tracker moving to another object at a collision does the same, so it isn't counted.",
                     jump * k, scale * k, 0.0, unit, float(z[i]))
    if z[i] >= Z_KINEMATIC and gaps[i] > practical:
        size = f" ({jump * a.px_per_m / a.object_size_px:.1f} object sizes)" if a.object_size_px else ""
        return Check("kinematics", title, "violation",
                     f"At {a.traj.t[i + 1]:.2f} s the object jumps {jump * k:.3g} {unit}{size} in one frame, beyond what "
                     f"its motion allows ({z[i]:.0f} times the noise). A tracker that hopped to a lookalike would do the same: "
                     "check the overlay.", jump * k, scale * k, 0.0, unit, float(z[i]))
    return Check("kinematics", title, "pass", "No jump beyond the noise between frames.", jump * k, scale * k, 0.0, unit, float(z[i]))


def check_morph(a: AuditInput, sigma: float = 0.0) -> Check:
    title = "No morphing"
    if a.lookalikes:
        return Check("kinematics", title, "skipped", "Not checked: objects like this one are in view, and touching "
                     "lookalikes merge into one mask.")
    if a.areas_px is None or len(a.areas_px) != len(a.traj) or np.any(np.asarray(a.areas_px) <= 0):
        return Check("kinematics", title, "skipped", "Not checked: mask areas aren't available for this track.")
    la = np.log(np.asarray(a.areas_px, dtype=float))
    m = MORPH_WINDOW
    if len(la) < 4 * m:
        return Check("kinematics", title, "skipped", "Not checked: the track is too short.")
    # Median area just after each frame against just before: a smooth change (moving towards the camera) shifts
    # every step a little; a morph is one step far larger than all the others.
    steps = np.array([np.median(la[i:i + m]) - np.median(la[i - m:i]) for i in range(m, len(la) - m + 1)])
    scale = max(_robust_scale(steps), 0.01)
    z = np.abs(steps - np.median(steps)) / scale
    i = int(np.argmax(z))
    factor = float(np.exp(steps[i]))
    if z[i] >= Z_KINEMATIC and abs(steps[i]) > np.log(MORPH_FACTOR) and \
            at_contact(a.traj.xy, a.traj.t, i + m - 1, i + m, sigma):
        return Check("kinematics", title, "skipped",
                     f"At {a.traj.t[i + m]:.2f} s the mask area changes {max(factor, 1 / factor):.1f} times, but at a contact "
                     "(the velocity changes there too): spray, splash or a second object touching it does the same, so it "
                     "isn't counted.", factor, None, 1.0, "", float(z[i]))
    if z[i] >= Z_KINEMATIC and abs(steps[i]) > np.log(MORPH_FACTOR):
        return Check("kinematics", title, "violation",
                     f"At {a.traj.t[i + m]:.2f} s the object's mask area {'grows' if factor > 1 else 'shrinks'} {max(factor, 1 / factor):.1f} "
                     f"times from one moment to the next ({z[i]:.0f} times the noise): its shape or size changes abruptly.",
                     factor, None, 1.0, "", float(z[i]))
    return Check("kinematics", title, "pass", "The object's mask keeps a steady size.", factor, None, 1.0, "", float(z[i]))


# ---------- the verdict ----------

def save_input(a: AuditInput, path: Path) -> None:
    """Everything the audit read, so a changed rule can be replayed on stored runs without another paid call."""
    arrays: dict[str, np.ndarray] = {"t": a.traj.t, "xy": a.traj.xy, "frames": a.traj.frames}
    for name in ("areas_px", "confidence", "reference_px"):
        value = getattr(a, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    meta = {k: getattr(a, k) for k in ("family", "calibration_source", "px_per_m", "scale_rel_std", "object_size_px",
                                       "coverage", "frame_rate_varies", "reference_size_px", "regime_change_s", "lookalikes")}
    meta["fps"] = a.traj.fps
    meta["law"] = asdict(a.law) if a.law is not None else None
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_input(path: Path) -> AuditInput:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    law = meta.pop("law")
    fps = meta.pop("fps")
    get = lambda k: z[k] if k in z.files else None  # noqa: E731
    return AuditInput(traj=Trajectory(z["t"], z["xy"], z["frames"], fps), law=LawEvidence(**law) if law else None,
                      areas_px=get("areas_px"), confidence=get("confidence"), reference_px=get("reference_px"), **meta)


def cant_judge(reason: str) -> dict[str, Any]:
    return {"verdict": "cant_judge", "headline": f"Can't judge: {reason}", "reasons": [reason], "checks": [],
            "violations": [], "thresholds": thresholds()}


def audit(a: AuditInput) -> dict[str, Any]:
    """Every applicable check, and the verdict: violations need preconditions met, a large z and a large magnitude."""
    if len(a.traj) < 3:
        return cant_judge("the object was tracked in too few frames.")
    sigma = noise_sigma(a.traj.xy)
    reasons = preconditions(a, sigma)
    if reasons:
        return {"verdict": "cant_judge", "headline": "Can't judge: " + " ".join(reasons), "reasons": reasons,
                "checks": [], "violations": [], "thresholds": thresholds()}

    checks: list[Check] = []
    family_seen = True
    if a.family == "projectile":
        flights = find_flights(a.traj, sigma)
        family_seen = bool(flights)
        checks.append(check_gravity_scale(a, flights))
        checks += check_gravity_consistency(a, flights)
        for c in (check_restitution(a, flights), check_flight_energy(a, flights)):
            if c is not None:
                checks.append(c)
        if not flights:
            checks.append(Check("energy", "Energy in free flight", "skipped", "Not checked: no clean free-fall flight was found."))
    elif a.family in ("pendulum", "oscillator"):
        if a.family == "pendulum":
            checks.append(check_gravity_scale(a, []))
        checks += check_oscillator(a)
        family_seen = not isinstance(oscillation_windows(a.traj, a.family), str)
    law = check_law(a, family_seen)
    if law is not None:
        checks.append(law)
    checks += [check_jumps(a, sigma), check_morph(a, sigma)]

    others = [c for c in checks if c.status == "violation" and c.check != "law"]
    for k, c in enumerate(checks):
        if c.check == "law" and c.status == "violation" and not others:
            # No law fitting is evidence about the model list as much as the video: several bodies colliding (a
            # Newton's cradle in the real control set) have no single law either. It accuses only alongside a
            # measured violation.
            checks[k] = Check("law", c.title, "skipped", c.text + " On its own that isn't proof: colliding or driven "
                              "objects have no single law either.", c.value, c.std, c.expected, c.unit, c.z)
    violations = [c for c in checks if c.status == "violation"]
    dynamic_passes = [c for c in checks if c.status == "pass" and c.check in DYNAMIC_CHECKS]
    if violations:
        verdict: Verdict = "violates_physics"
        headline = "Violates physics: " + "; ".join(c.text for c in violations)
    elif dynamic_passes:
        verdict = "consistent"
        headline = (f"Consistent with physics: {len(dynamic_passes)} dynamics check{'s' if len(dynamic_passes) > 1 else ''} "
                    "passed and nothing jumps or morphs.")
    else:
        # Smooth motion alone proves little: without one dynamics check there is nothing to vouch for.
        verdict = "cant_judge"
        headline = (f"Can't judge: no dynamics check applies to {a.family} motion here, and smooth tracking alone "
                    "doesn't show the motion is physical.")
    return {
        "verdict": verdict, "headline": headline, "reasons": [] if verdict != "cant_judge" else [headline],
        "checks": [asdict(c) for c in checks], "violations": [asdict(c) for c in violations], "thresholds": thresholds(),
    }


def thresholds() -> dict[str, float]:
    return {"z": Z_VIOLATION, "z_kinematic": Z_KINEMATIC, "g_rel": G_REL_TOL, "flight_g_rel": FLIGHT_G_REL_TOL,
            "max_restitution": MAX_RESTITUTION, "amplitude_growth": AMPLITUDE_GROWTH, "period_drift": PERIOD_DRIFT,
            "no_law_nrmse": NO_LAW_NRMSE}
