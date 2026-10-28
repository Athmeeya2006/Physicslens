"""The closed experiment loop: a small swing ties, the larger swing the agent asks for separates the laws."""

import numpy as np
import pytest
from physicslens.physics.coords import to_coordinates
from physicslens.physics.fit import fit_model
from physicslens.physics.joint import decide_joint, fit_joint
from physicslens.physics.models import MODELS
from physicslens.physics.verify import assess
from physicslens.sigproc.trajectory import Trajectory
from scipy.integrate import solve_ivp

from .test_physics import FPS, G

L = 0.6
RIVALS = ("pendulum_linear_damping", "pendulum_quadratic_damping")


def air_drag_pendulum(theta0: float, c: float, seconds: float, seed: int):
    """Quadratic (air) damping: it barely shows at small amplitude, where it looks like linear damping."""
    t = np.arange(0, seconds, 1 / FPS)
    th = solve_ivp(lambda _t, s: [s[1], -G / L * np.sin(s[0]) - c * s[1] * abs(s[1])], (0, seconds), [theta0, 0],
                   t_eval=t, rtol=1e-10, atol=1e-12).y[0]
    xy = np.column_stack([0.1 + L * np.sin(th), 1.0 - L * np.cos(th)])
    xy += np.random.default_rng(seed).normal(0, 0.002, xy.shape)
    return to_coordinates(Trajectory(t, xy, np.arange(len(t)), FPS), "pendulum")


def test_small_swing_ties_and_the_requested_large_swing_separates_the_laws():
    small = air_drag_pendulum(0.12, 0.25, 8, seed=1)
    alone = assess(small, [fit_model(MODELS[m], small) for m in RIVALS])
    assert alone.status == "ambiguous"  # the premise: one small clip can't tell them apart
    large = air_drag_pendulum(1.0, 0.25, 8, seed=2)
    verdict = decide_joint([fit_joint(MODELS[m], [small, large]) for m in RIVALS])
    assert verdict["status"] == "accepted" and verdict["winner"].model_id == "pendulum_quadratic_damping"
    assert verdict["winner"].params["beta"] == pytest.approx(0.25, rel=0.15)
    loser = next(f for f in verdict["ranked"] if f.model_id == "pendulum_linear_damping")
    assert loser.bic - verdict["winner"].bic > 10


def test_joint_fit_needs_every_clip_forecast():
    small = air_drag_pendulum(0.5, 0.25, 8, seed=3)
    fit = fit_joint(MODELS["pendulum_quadratic_damping"], [small, small])
    assert fit.success and fit.forecasts_well and len(fit.heldout_nrmse) == 2


def test_study_review_fits_the_recordings_together(monkeypatch):
    import physicslens.export.sim2real as s2r
    from physicslens.studies import joint_verdict

    clips = {"small": air_drag_pendulum(0.12, 0.25, 8, seed=1), "large": air_drag_pendulum(1.0, 0.25, 8, seed=2)}
    monkeypatch.setattr(s2r, "measured_coordinates", lambda res: clips[res["clip"]])
    ranking = [{"model_id": m} for m in RIVALS]
    done = [("Small swing", "r1", {"clip": "small", "status": "ambiguous", "family": "pendulum", "ranking": ranking,
                                   "fit_state": {"model_id": RIVALS[0]}}),
            ("Wider swing", "r2", {"clip": "large", "status": "accepted", "family": "pendulum", "ranking": ranking,
                                   "fit_state": {"model_id": "pendulum_quadratic_damping"}})]
    out = joint_verdict(done)
    assert out["status"] == "accepted" and "air" in out["text"].lower() and "Small swing" in out["text"]
    # Nothing tied, nothing to settle: no joint fit.
    assert joint_verdict([(lab, rid, {**res, "status": "accepted"}) for lab, rid, res in done]) is None
