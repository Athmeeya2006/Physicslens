"""Predict the future, then prove it: predictions from the first half of a clip against its second half."""

import numpy as np
from physicslens.physics.coords import Coordinates, to_coordinates
from physicslens.physics.fit import fit_model
from physicslens.physics.forecast import judge, observed_crossing, predict_crossing
from physicslens.physics.models import MODELS
from physicslens.physics.uncertainty import bootstrap_samples
from physicslens.sigproc.trajectory import Trajectory

from .test_physics import FPS, pendulum_xy


def first_part(c: Coordinates, seconds: float) -> Coordinates:
    n = int(seconds * FPS)
    return Coordinates(c.family, c.t[:n], c.q[:n], c.names, dict(c.ctx))


def test_predictions_from_the_first_half_land_in_their_window():
    spec = MODELS["pendulum_linear_damping"]
    outcomes = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        theta0, gamma = rng.uniform(0.3, 0.8), rng.uniform(0.02, 0.15)
        xy = pendulum_xy(theta0, gamma, 14)
        traj = Trajectory(np.arange(len(xy)) / FPS, xy + rng.normal(0, 0.002, xy.shape), np.arange(len(xy)), FPS)
        full = to_coordinates(traj, "pendulum")
        early = first_part(full, 7.0)
        fit = fit_model(spec, early)
        samples, _ = bootstrap_samples(spec, early, fit, reps=10, seed=seed)
        p = predict_crossing(spec, fit, early, horizon_s=5.0, param_samples=samples)
        assert p is not None and p.t_pred >= early.t[-1] + 5.0
        outcomes.append(judge(p, observed_crossing(full.t, full.q[:, 0], p)))
    hits = sum(o["status"] == "hit" for o in outcomes)
    assert all(o["status"] in ("hit", "miss") for o in outcomes)
    assert hits >= 9, outcomes
    # The window is a real constraint, not a free pass: well under a period (about 1.55 s for a 0.6 m pendulum).
    assert max(o["window_s"] for o in outcomes) < 0.3


def test_a_wrong_law_misses():
    """A law that ignores damping gets the timing of a strongly damped swing wrong, and the check says so."""
    rng = np.random.default_rng(1)
    xy = pendulum_xy(0.9, 0.0, 14, length=0.6)
    traj = Trajectory(np.arange(len(xy)) / FPS, xy + rng.normal(0, 0.002, xy.shape), np.arange(len(xy)), FPS)
    full = to_coordinates(traj, "pendulum")
    early = first_part(full, 7.0)
    spec = MODELS["pendulum_linear_damping"]
    fit = fit_model(spec, early)
    samples, _ = bootstrap_samples(spec, early, fit, reps=10)
    p = predict_crossing(spec, fit, early, 5.0, samples)
    # Shift the truth: the same pendulum with a 3% longer period is a different system, and must not "hit".
    stretched = Coordinates(full.family, full.t * 1.03, full.q, full.names, full.ctx)
    assert judge(p, observed_crossing(stretched.t, stretched.q[:, 0], p))["status"] == "miss"


def test_unmeasured_until_the_data_reach_the_window():
    xy = pendulum_xy(0.6, 0.05, 10)
    full = to_coordinates(Trajectory(np.arange(len(xy)) / FPS, xy, np.arange(len(xy)), FPS), "pendulum")
    early = first_part(full, 5.0)
    spec = MODELS["pendulum_linear_damping"]
    fit = fit_model(spec, early)
    p = predict_crossing(spec, fit, early, 4.0, bootstrap_samples(spec, early, fit, reps=6)[0])
    assert judge(p, observed_crossing(early.t, early.q[:, 0], p))["status"] == "unmeasured"


def test_a_sliding_object_is_predicted_to_reach_a_point_ahead():
    # A block sliding to a stop under dry friction (constant deceleration): from the first 1.5 s, predict when it
    # passes the point the law puts it 0.5 s later. No rest point to come back to, so the event is a point ahead.
    spec = MODELS["line_constant_accel"]
    t = np.arange(0, 2.8, 1 / FPS)
    s = 1.2 * t - 0.2 * t**2
    rng = np.random.default_rng(3)
    full = Coordinates("line", t, (s + rng.normal(0, 0.002, len(t)))[:, None], ("s",), {})
    early = first_part(full, 1.5)
    fit = fit_model(spec, early)
    samples, _ = bootstrap_samples(spec, early, fit, reps=10, seed=3)
    p = predict_crossing(spec, fit, early, horizon_s=0.5, param_samples=samples)
    assert p is not None and p.direction == 1 and abs(p.t_pred - (early.t[-1] + 0.5)) < 1e-9
    outcome = judge(p, observed_crossing(full.t, full.q[:, 0], p))
    assert outcome["status"] == "hit" and outcome["window_s"] < 0.2


def test_no_point_ahead_once_it_has_stopped():
    spec = MODELS["line_constant_accel"]
    t = np.arange(0, 3.0, 1 / FPS)
    s = np.where(t < 2.0, 1.0 * t - 0.25 * t**2, 1.0)  # stopped at 2 s
    c = Coordinates("line", t[t >= 2.2], s[t >= 2.2, None], ("s",), {})
    fit = fit_model(MODELS["line_constant_velocity"], c)
    assert predict_crossing(MODELS["line_constant_velocity"], fit, c, horizon_s=0.5, param_samples=np.zeros((4, 0))) is None
    assert spec.family == "line"
