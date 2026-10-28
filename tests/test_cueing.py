"""Control by hand: a simulated person following the rhythm cues pumps a swing up to a target, or stops it."""

import numpy as np
from physicslens.physics.coords import Coordinates
from physicslens.physics.cueing import learned_lead, next_cue, push_delay
from physicslens.physics.fit import fit_model
from physicslens.physics.models import MODELS
from scipy.integrate import solve_ivp

FPS, G, L, GAMMA = 30, 9.81, 0.6, 0.06
SPEC = MODELS["pendulum_linear_damping"]


class Swing:
    """The real system: a damped pendulum the hand can kick, watched by a camera with 2 mrad of noise."""

    def __init__(self, theta0: float, seed: int):
        self.rng = np.random.default_rng(seed)
        self.t, self.state = 0.0, np.array([theta0, 0.0])
        self.ts, self.th = [0.0], [theta0]
        self.last_push = -1.0

    def run_until(self, t_end: float) -> None:
        t_eval = np.arange(self.t + 1 / FPS, t_end, 1 / FPS)
        t_eval = t_eval[t_eval < t_end]  # arange can overshoot its stop by a rounding error
        sol = solve_ivp(lambda _t, s: [s[1], -G / L * np.sin(s[0]) - GAMMA * s[1]], (self.t, t_end), self.state,
                        t_eval=t_eval, rtol=1e-9, atol=1e-11, dense_output=True)
        self.ts += list(sol.t)
        self.th += list(sol.y[0])
        self.t, self.state = t_end, sol.sol(t_end)

    def push(self, dv: float) -> None:
        self.state = self.state + np.array([0.0, dv])
        self.last_push = self.t

    def observed(self, seconds: float) -> Coordinates:
        """Only motion since the last push: a push is outside the law, so a fit across it would be wrong."""
        ts, th = np.array(self.ts), np.array(self.th)
        keep = (ts >= ts[-1] - seconds) & (ts > self.last_push + 0.1)
        noisy = th[keep] + self.rng.normal(0, 0.002, keep.sum())
        return Coordinates("pendulum", ts[keep] - ts[keep][0], noisy[:, None], ("theta",),
                           {"length_m": L, "pivot_x": 0.0, "pivot_y": 0.0})

    def amplitude(self, seconds: float = 2.0) -> float:
        ts, th = np.array(self.ts), np.array(self.th)
        return float(np.max(np.abs(th[ts >= ts[-1] - seconds])))


def follow_cues(goal: str, theta0: float, target: float, kick: float, seed: int, max_pushes: int = 10,
                learn_lag: bool = True) -> tuple[int, Swing, list[float]]:
    """A person who pushes in the cued direction about 0.22 s late (sd 0.05 s), with a varying push strength.

    The app measures each push's real time from the track and cues later pushes that much earlier.
    """
    swing = Swing(theta0, seed)
    swing.run_until(6.0)
    pushes, delays = 0, []
    while pushes < max_pushes:
        c = swing.observed(6.0)
        fit = fit_model(SPEC, c)
        lead = learned_lead(delays) if learn_lag else 0.0
        cue = next_cue(SPEC, fit, c, goal, target, lead_s=lead)
        assert cue is not None
        if cue.action in ("done", "wait"):
            return pushes, swing, delays
        offset = swing.t - float(c.t[-1])  # coordinates' clock to the swing's clock
        first_push_at = None
        for t_push in cue.pushes:
            at = t_push + offset + swing.rng.normal(0.22, 0.05)
            if at <= swing.t or pushes >= max_pushes:
                continue
            swing.run_until(at)
            # To stop it, a person matches the push to the swing rather than shoving as hard as they can.
            strength = kick if goal == "pump" else min(kick, cue.bottom_speed)
            swing.push(cue.direction * strength * swing.rng.uniform(0.7, 1.3))
            pushes += 1
            first_push_at = first_push_at or t_push
        swing.run_until(swing.t + 2.5)  # about a period and a half of free motion to refit on
        if first_push_at is not None:
            ts, th = np.array(swing.ts), np.array(swing.th)
            seen = ts - offset > c.t[0]
            t_obs = ts[seen] - offset
            q_obs = th[seen] + swing.rng.normal(0, 0.002, seen.sum())
            d = push_delay(SPEC, fit, c, t_obs, q_obs, first_push_at)
            if d is not None:
                delays.append(d + lead)  # how late the push lands after a cue placed at the bottom itself
    c = swing.observed(6.0)
    cue = next_cue(SPEC, fit_model(SPEC, c), c, goal, target)
    return (pushes if cue is not None and cue.action in ("done", "wait") else pushes + 1), swing, delays


def test_rhythm_cues_pump_a_swing_to_the_target_within_ten_pushes():
    pushes, swing, _ = follow_cues("pump", theta0=0.09, target=np.radians(30), kick=0.45, seed=1)
    assert pushes <= 10
    assert abs(swing.amplitude() - np.radians(30)) < np.radians(4)


def test_without_learning_the_lag_stopping_stalls():
    """The lag matters: pushes that land 0.22 s after the bottom leave most of the swing's height in place."""
    pushes, swing, _ = follow_cues("stop", theta0=0.6, target=0.05, kick=0.7, seed=2, learn_lag=False)
    assert pushes > 10 or swing.amplitude() > 0.08


def test_rhythm_cues_stop_a_swing():
    pushes, swing, delays = follow_cues("stop", theta0=0.6, target=0.05, kick=0.7, seed=2)
    assert pushes <= 10 and swing.amplitude() < 0.08
    # The app measured the person's lag from the track itself.
    assert abs(learned_lead(delays) - 0.22) < 0.08
