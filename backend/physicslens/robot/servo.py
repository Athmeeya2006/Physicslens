"""Joint-level laws for position-controlled hobby servos, fitted to logged command and measured angle.

Pure functions, no network. Units: degrees and seconds. Every law is simulated open loop from the command alone
(starting at the first measured angle, at rest), never one step ahead from the measured angle: a simulator replaying
logged commands has no measurement to lean on, so open-loop error is the error a user of the model would see.

With u the command and theta the measured angle, every law drives on the tracking error
    e = u(t - d) + b [+ sc cos(theta) + ss sin(theta)] - theta
and is one of
    lag1    theta' = e / tau                                   (first-order lag)
    so2     theta'' = wn^2 e - 2 zeta wn theta'                (second-order servo)
with optional features:
    rate      |theta'| <= wmax                                 (speed limit)
    coulomb   - fc sign(theta') in the acceleration            (dry friction, smoothed sign)
    deadband  e replaced by sign(e) max(|e| - db, 0)           (deadband or backlash: no drive inside +-db)
    sag       the bracketed angle-dependent offset             (gravity sag or calibration mismatch that varies
                                                                with the angle; these data cannot tell which)
The constant b absorbs the calibration offset between the arm that commands (the leader) and the arm that moves
(the follower); it is reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

# Integration substeps per logged frame: at 30 fps this is 3.3 ms, small next to the fastest time constant the
# bounds allow (tau >= 15 ms; wn <= 120 rad/s keeps h * wn <= 0.4 for the semi-implicit Euler step).
SUBSTEPS = 10
# Smoothing width of the friction sign (deg/s): a hard sign makes the least-squares cost non-differentiable.
SIGN_EPS = 2.0

BOUNDS = {"tau": (0.015, 1.0), "wn": (2.0, 120.0), "zeta": (0.05, 6.0), "d": (0.0, 0.4), "b": (-20.0, 20.0),
          "wmax": (5.0, 2000.0), "fc": (0.0, 1e5), "db": (0.0, 10.0), "sc": (-20.0, 20.0), "ss": (-20.0, 20.0)}
FEATURE_PARAMS = {"rate": ("wmax",), "coulomb": ("fc",), "deadband": ("db",), "sag": ("sc", "ss")}
FEATURE_TEXT = {"rate": "speed limit", "coulomb": "dry friction", "deadband": "deadband", "sag": "angle-dependent offset"}


@dataclass(frozen=True)
class Law:
    id: str
    base: str  # "lag1" or "so2"
    features: tuple[str, ...]

    @property
    def params(self) -> tuple[str, ...]:
        core = ("tau",) if self.base == "lag1" else ("wn", "zeta")
        return core + ("d", "b") + tuple(p for f in self.features for p in FEATURE_PARAMS[f])

    @property
    def lower(self) -> tuple[float, ...]:
        return tuple(BOUNDS[p][0] for p in self.params)

    @property
    def upper(self) -> tuple[float, ...]:
        return tuple(BOUNDS[p][1] for p in self.params)

    @property
    def title(self) -> str:
        head = "first-order lag" if self.base == "lag1" else "second-order servo"
        return ", ".join([f"{head} with delay", *(FEATURE_TEXT[f] for f in self.features)])


def _law(law_id: str) -> Law:
    base, *feats = law_id.split("_")
    return Law(law_id, base, tuple(feats))


LAWS: dict[str, Law] = {i: _law(i) for i in [
    "lag1", "lag1_rate", "lag1_deadband", "lag1_sag", "lag1_deadband_sag",
    "so2", "so2_rate", "so2_coulomb", "so2_deadband", "so2_sag", "so2_deadband_sag",
]}
# Each richer law and the simpler laws nested in it. A richer law is chosen only if it beats its parent on
# held-out episodes (see select()); the search walks down from lag1.
CHILDREN = {
    "lag1": ("lag1_rate", "lag1_deadband", "lag1_sag", "so2"),
    "lag1_deadband": ("lag1_deadband_sag", "so2_deadband"),
    "lag1_sag": ("lag1_deadband_sag", "so2_sag"),
    "so2": ("so2_rate", "so2_coulomb", "so2_deadband", "so2_sag"),
    "so2_deadband": ("so2_deadband_sag",),
    "so2_sag": ("so2_deadband_sag",),
    "lag1_deadband_sag": ("so2_deadband_sag",),
}


@dataclass
class Batch:
    """Episodes of one joint, padded to a common length; mask marks real frames."""
    u: np.ndarray  # [E, N] command (deg)
    y: np.ndarray  # [E, N] measured angle (deg)
    mask: np.ndarray  # [E, N] bool
    dt: float
    ids: list[int] = field(default_factory=list)

    @staticmethod
    def from_lists(us: list[np.ndarray], ys: list[np.ndarray], dt: float, ids: list[int] | None = None) -> Batch:
        n = max(len(u) for u in us)
        e = len(us)
        u = np.zeros((e, n))
        y = np.zeros((e, n))
        m = np.zeros((e, n), bool)
        for i, (ui, yi) in enumerate(zip(us, ys, strict=True)):
            k = len(ui)
            # Padding repeats the last sample so the simulated state stays finite; the mask drops it from every error.
            u[i, :k], u[i, k:] = ui, ui[-1]
            y[i, :k], y[i, k:] = yi, yi[-1]
            m[i, :k] = True
        return Batch(u, y, m, dt, list(ids) if ids is not None else list(range(e)))

    def take(self, idx: np.ndarray | list[int]) -> Batch:
        idx = np.asarray(idx)
        return Batch(self.u[idx], self.y[idx], self.mask[idx], self.dt, [self.ids[i] for i in idx])


def delayed_input(u: np.ndarray, d: float, dt: float, substeps: int) -> np.ndarray:
    """u(t - d) at every substep time, linearly interpolated; before the log starts the first command holds."""
    n = u.shape[1]
    t = np.arange(n * substeps) * (dt / substeps)
    x = np.clip((t - d) / dt, 0.0, n - 1.0)
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    f = x - i0
    return u[:, i0] * (1.0 - f) + u[:, i0 + 1] * f


def simulate(law_id: str, p: dict[str, float], batch: Batch, substeps: int = SUBSTEPS) -> np.ndarray:
    """Open-loop angle at every logged frame [E, N], from the first measured angle at rest."""
    law = LAWS[law_id]
    f = set(law.features)
    h = batch.dt / substeps
    ud = delayed_input(batch.u, p["d"], batch.dt, substeps) + p["b"]
    th = batch.y[:, 0].copy()
    v = np.zeros_like(th)
    out = np.empty_like(batch.y)
    wmax = p.get("wmax", np.inf)
    for k in range(ud.shape[1]):
        if k % substeps == 0:
            out[:, k // substeps] = th
        e = ud[:, k] - th
        if "sag" in f:
            r = np.deg2rad(th)
            e = e + p["sc"] * np.cos(r) + p["ss"] * np.sin(r)
        if "deadband" in f:
            e = np.sign(e) * np.maximum(np.abs(e) - p["db"], 0.0)
        if law.base == "lag1":
            th = th + h * np.clip(e / p["tau"], -wmax, wmax)
            continue
        wn, z = p["wn"], p["zeta"]
        drive = wn * wn * e
        if "coulomb" in f:
            drive = drive - p["fc"] * np.tanh(v / SIGN_EPS)
        # Semi-implicit Euler with the damping taken implicitly, so a stiff, well-damped servo stays stable.
        v = np.clip((v + h * drive) / (1.0 + h * 2.0 * z * wn), -wmax, wmax)
        th = th + h * v
    return out


def residuals(law_id: str, p: dict[str, float], batch: Batch) -> np.ndarray:
    return (simulate(law_id, p, batch) - batch.y)[batch.mask]


def rms_per_episode(sim: np.ndarray, batch: Batch) -> np.ndarray:
    """Root-mean-square error of a simulated [E, N] angle against the measured one, one value per episode."""
    err = (sim - batch.y) ** 2
    return np.sqrt((err * batch.mask).sum(1) / batch.mask.sum(1))


def best_shift(batch: Batch, max_shift: int = 10) -> int:
    """Whole-frame shift of the command that best matches the measured angle (the delay a user would read off)."""
    errs = []
    for s in range(max_shift):
        m = batch.mask[:, s:] & batch.mask[:, : batch.mask.shape[1] - s]
        errs.append(np.mean(((batch.y[:, s:] - batch.u[:, : batch.u.shape[1] - s]) ** 2)[m]))
    return int(np.argmin(errs))


def initial_guess(law_id: str, batch: Batch) -> dict[str, float]:
    """Delay from the best shift, offset from the mean gap, a speed limit at the fastest motion seen."""
    d = batch.dt * best_shift(batch)
    b = float(np.mean((batch.y - batch.u)[batch.mask]))
    speed = np.abs(np.diff(batch.y, axis=1))[batch.mask[:, 1:]] / batch.dt
    g = {"tau": 0.06, "d": max(d - 0.05, 0.0), "b": b, "wn": 40.0, "zeta": 1.5, "fc": 50.0, "db": 0.3,
         "wmax": float(np.percentile(speed, 99.9)) if speed.size else 300.0, "sc": 0.0, "ss": 0.0}
    return {k: g[k] for k in LAWS[law_id].params}


@dataclass(frozen=True)
class Fit:
    law_id: str
    params: dict[str, float]
    rss: float
    n: int
    bic: float


def _clip_into(law: Law, x: np.ndarray) -> np.ndarray:
    lo, hi = np.array(law.lower), np.array(law.upper)
    return np.clip(x, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))


def fit(law_id: str, batch: Batch, start: dict[str, float] | None = None, weights: np.ndarray | None = None,
        max_nfev: int = 200) -> Fit:
    """Least-squares fit of one law to every frame of the given episodes (weights = per-episode multiplicity)."""
    law = LAWS[law_id]
    s = initial_guess(law_id, batch) | (start or {})
    x0 = _clip_into(law, np.array([s[k] for k in law.params]))
    w = np.ones(len(batch.ids)) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(np.broadcast_to(w[:, None], batch.y.shape))[batch.mask]

    def r(x: np.ndarray) -> np.ndarray:
        return residuals(law_id, dict(zip(law.params, x, strict=True)), batch) * sw

    sol = least_squares(r, x0, bounds=(law.lower, law.upper), x_scale="jac", max_nfev=max_nfev)
    p = dict(zip(law.params, map(float, sol.x), strict=True))
    res = residuals(law_id, p, batch)
    rss = float(res @ res)
    return Fit(law_id, p, rss, int(res.size), bic(rss, int(res.size), len(law.params)))


def bic(rss: float, n: int, k: int) -> float:
    """Gaussian BIC. Frames within an episode are correlated, so this overstates the evidence for extra terms;
    that is why held-out episodes, not BIC alone, decide (select())."""
    return n * float(np.log(rss / n)) + k * float(np.log(n))


def bootstrap(law_id: str, batch: Batch, best: dict[str, float], reps: int, seed: int = 0) -> np.ndarray:
    """Refits on episodes drawn with replacement [reps, k]. Whole episodes are the honest unit of repetition:
    frames inside one episode are strongly correlated."""
    rng = np.random.default_rng(seed)
    e = len(batch.ids)
    out = []
    for _ in range(reps):
        counts = np.bincount(rng.integers(0, e, e), minlength=e)
        keep = np.flatnonzero(counts)
        f = fit(law_id, batch.take(keep), start=best, weights=counts[keep], max_nfev=40)
        out.append([f.params[k] for k in LAWS[law_id].params])
    return np.array(out)


def mean_interval(x: np.ndarray, reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean over episodes and its 95% bootstrap interval (resampling episodes)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@dataclass(frozen=True)
class Step:
    parent: str
    child: str
    gain: tuple[float, float, float]  # held-out RMS of parent minus child: mean, 95% interval
    delta_bic: float  # child minus parent, on training episodes
    accepted: bool


def select(fits: dict[str, Fit], heldout_rms: dict[str, np.ndarray]) -> tuple[str, list[Step]]:
    """Walk from lag1 to richer laws. A richer law replaces its parent only when it lowers the held-out RMS with the
    95% interval of the paired gain (per held-out episode) above zero AND it lowers BIC on the training episodes.
    Among children that pass, the one with the lowest held-out RMS is taken; the walk stops when none pass."""
    cur, steps = "lag1", []
    while True:
        passing = []
        for c in CHILDREN.get(cur, ()):
            if c not in fits:
                continue
            g = mean_interval(heldout_rms[cur] - heldout_rms[c])
            db = fits[c].bic - fits[cur].bic
            ok = g[1] > 0.0 and db < 0.0
            steps.append(Step(cur, c, g, db, ok))
            if ok:
                passing.append(c)
        if not passing:
            return cur, steps
        cur = min(passing, key=lambda c: float(heldout_rms[c].mean()))
