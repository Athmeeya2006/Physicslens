"""Synthetic trajectories with known equations: the fitting and verification must recover them."""

import numpy as np
import pytest
from physicslens.physics.coords import length_is_reliable, to_coordinates
from physicslens.physics.fit import fit_model
from physicslens.physics.models import MODELS, models_for
from physicslens.physics.sindy import discover, vet
from physicslens.physics.uncertainty import bootstrap
from physicslens.physics.verify import assess, judge_candidates
from physicslens.sigproc.trajectory import Trajectory
from scipy.integrate import solve_ivp

FPS = 30
NOISE_M = 0.002  # ~2 px at 1000 px/m, typical for SAM centroids
G = 9.81


def track(xy: np.ndarray, seed: int = 1) -> Trajectory:
    rng = np.random.default_rng(seed)
    t = np.arange(len(xy)) / FPS
    return Trajectory(t, xy + rng.normal(0, NOISE_M, xy.shape), np.arange(len(t)), FPS)


def pendulum_xy(theta0: float, gamma: float, seconds: float, length: float = 0.6) -> np.ndarray:
    t = np.arange(0, seconds, 1 / FPS)
    th = solve_ivp(lambda _t, s: [s[1], -G / length * np.sin(s[0]) - gamma * s[1]], (0, seconds), [theta0, 0],
                   t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    return np.column_stack([0.1 + length * np.sin(th), 1.0 - length * np.cos(th)])


def run(traj: Trajectory, family: str):
    c = to_coordinates(traj, family)
    return c, assess(c, [fit_model(m, c) for m in models_for(family)])


@pytest.fixture(scope="module")
def damped_pendulum():
    return run(track(pendulum_xy(0.9, 0.08, 10)), "pendulum")


def test_damped_pendulum_recovers_g_and_damping(damped_pendulum):
    _, v = damped_pendulum
    assert v.status == "accepted" and v.winner.model_id == "pendulum_linear_damping"
    assert v.winner.params["g"] == pytest.approx(G, rel=0.03)
    assert v.winner.params["gamma"] == pytest.approx(0.08, rel=0.1)


def test_bootstrap_interval_covers_truth(damped_pendulum):
    c, v = damped_pendulum
    est = bootstrap(MODELS[v.winner.model_id], c, v.winner, reps=12)
    assert est["g"].ci_low - 3 * est["g"].std <= G <= est["g"].ci_high + 3 * est["g"].std
    assert 0 < est["g"].std < 0.3


def test_sindy_finds_pendulum_structure(damped_pendulum):
    c, _ = damped_pendulum
    terms = discover(c).terms["theta''"]
    assert terms["sin(theta)"] == pytest.approx(-G / c.ctx["length_m"], rel=0.05)


def test_sindy_is_shown_when_it_forecasts_like_the_verified_law(damped_pendulum):
    c, v = damped_pendulum
    out = vet(c, v.winner)
    assert out["shown"], out["reason"]


def test_sindy_is_hidden_when_its_forecast_is_far_worse():
    rng = np.random.default_rng(3)
    xy = pendulum_xy(0.9, 0.08, 10)
    c, v = run(Trajectory(np.arange(len(xy)) / FPS, xy + rng.normal(0, 0.02, xy.shape), np.arange(len(xy)), FPS),
               "pendulum")
    out = vet(c, v.winner)
    # Noisy derivatives push the sparse fit off the law; it stays in the JSON but is not shown.
    assert out["equations"] and not out["shown"] and out["reason"]


def test_sindy_is_hidden_without_a_verified_law(damped_pendulum):
    c, _ = damped_pendulum
    assert vet(c, None)["shown"] is False


def test_undamped_pendulum_prefers_simplest_model():
    _, v = run(track(pendulum_xy(0.7, 0.0, 6)), "pendulum")
    assert v.status == "accepted" and v.winner.model_id == "pendulum"
    assert v.winner.params["g"] == pytest.approx(G, rel=0.03)


def test_flat_arc_refuses_to_report_g():
    c, v = run(track(pendulum_xy(0.08, 0.0, 6)), "pendulum")
    assert not length_is_reliable(c)
    assert v.status == "ambiguous" and v.winner is None


def test_damped_spring():
    t = np.arange(0, 8, 1 / FPS)
    s = 0.05 * np.exp(-0.15 * t) * np.cos(2 * np.pi * 1.3 * t)
    _, v = run(track(np.column_stack([0.01 * s, s + 0.3])), "oscillator")
    assert v.status == "accepted" and v.winner.model_id == "damped_harmonic"
    assert v.winner.params["omega"] == pytest.approx(2 * np.pi * 1.3, rel=0.01)
    assert v.winner.params["zeta"] * v.winner.params["omega"] == pytest.approx(0.15, rel=0.1)


def test_projectile_with_quadratic_drag():
    secs = 0.9
    t = np.arange(0, secs, 1 / FPS)

    def rhs(_t, s):
        speed = np.hypot(s[2], s[3])
        return [s[2], s[3], -0.4 * speed * s[2], -G - 0.4 * speed * s[3]]

    xy = solve_ivp(rhs, (0, secs), [0, 0, 2, 3], t_eval=t, rtol=1e-10).y[:2].T
    _, v = run(track(xy), "projectile")
    assert v.status == "accepted" and v.winner.model_id == "quadratic_drag"
    assert v.winner.params["g"] == pytest.approx(G, rel=0.03)


def test_random_motion_has_no_clean_equation():
    rng = np.random.default_rng(3)
    _, v = run(track(np.cumsum(rng.normal(0, 0.01, (240, 2)), axis=0)), "oscillator")
    assert v.status == "no_fit" and v.winner is None


def test_coordinates_map_back_to_xy(damped_pendulum):
    c, v = damped_pendulum
    xy = c.to_xy(v.winner.sim)
    truth = pendulum_xy(0.9, 0.08, 10)
    assert np.sqrt(np.mean((xy - truth) ** 2)) < 3 * NOISE_M


def test_trail_explains_each_rejection(damped_pendulum):
    _, v = damped_pendulum
    judged = judge_candidates(v)
    assert judged["pendulum_linear_damping"][0] == "accepted"
    assert judged["pendulum"][0] == "rejected" and "forecast error" in judged["pendulum"][1]
    assert sum(st == "accepted" for st, _ in judged.values()) == 1


def test_tie_leads_to_a_designed_experiment():
    from physicslens.physics.design import design_experiment
    c, v = run(track(pendulum_xy(0.12, 0.02, 6.5, length=0.5)), "pendulum")
    assert v.status == "ambiguous" and len(v.contenders) >= 2
    d = design_experiment(c, list(v.contenders))
    assert d["decisive"] and d["recommended"]["separation_vs_noise"] >= 5
    # A larger swing is what separates damping laws; the current tiny swing alone would not.
    assert "degrees" in d["recommended"]["label"] and "7 degrees" not in d["recommended"]["label"]


def test_no_design_without_a_tie(damped_pendulum):
    from physicslens.physics.design import design_experiment
    c, v = damped_pendulum
    assert design_experiment(c, list(v.contenders)) is None  # accepted: a single contender


def test_small_angle_approximation_is_not_a_rival_theory():
    """A small swing can't separate sin(theta) from theta: keep the exact law, note the approximation, no experiment."""
    _, verdict = run(track(pendulum_xy(0.3, 0.0, 6.0, length=0.6), seed=5), "pendulum")
    assert verdict.status == "accepted" and verdict.winner.model_id == "pendulum"
    assert "indistinguishable" in verdict.reason
    assert verdict.winner.params["g"] == pytest.approx(G, rel=0.03)
    why = judge_candidates(verdict)["pendulum_small_angle"]
    assert why[0] == "approximation" and "exact law is kept" in why[1]


def test_dry_friction_spring_is_told_from_viscous_damping():
    """Coulomb friction removes the same amplitude every swing (straight-line decay); viscous damping doesn't."""
    from physicslens.physics.models import MODELS as M

    t = np.arange(0, 6, 1 / FPS)
    y = solve_ivp(lambda _t, s: M["oscillator_dry"].rhs(s, {"omega": 6.0, "s0": 0.0, "f": 0.12}, {}), (0, 6), [0.08, 0.0],
                  t_eval=t, rtol=1e-9, atol=1e-11, max_step=0.005).y[0]
    _, v = run(track(np.column_stack([0.01 * y, y + 0.3])), "oscillator")
    assert v.status == "accepted" and v.winner.model_id == "oscillator_dry"
    assert v.winner.params["f"] == pytest.approx(0.12, rel=0.1)


def test_door_closer_is_a_damped_torsional_spring():
    from physicslens.physics.coords import Coordinates

    t = np.arange(0, 5, 1 / FPS)
    true = {"kappa": 9.0, "phi0": 0.2, "k": 0.8}
    phi = solve_ivp(lambda _t, s: MODELS["spin_spring"].rhs(s, true, {}), (0, 5), [1.4, 0.0], t_eval=t, rtol=1e-10).y[0]
    phi = phi + np.random.default_rng(3).normal(0, 0.004, len(t))
    c = Coordinates("rotation", t, phi[:, None], ("phi",), {"arm_m": 0.3, "center_x": 0.0, "center_y": 0.0})
    v = assess(c, [fit_model(m, c) for m in models_for("rotation")])
    assert v.status == "accepted" and v.winner.model_id == "spin_spring"
    assert v.winner.params["kappa"] == pytest.approx(9.0, rel=0.03) and v.winner.params["k"] == pytest.approx(0.8, rel=0.1)
