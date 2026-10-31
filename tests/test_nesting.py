"""A law never ends worse than a special case it contains (ModelSpec.nests), and the guard says so if it would."""
import numpy as np
import pytest
from physicslens.physics.coords import Coordinates
from physicslens.physics.fit import (
    NESTED_REL_TOL,
    FitResult,
    _nesting_guard,
    fit_family,
    fit_model,
    simulate,
)
from physicslens.physics.models import BUILTIN_MODELS, NESTINGS
from physicslens.physics.verify import Verdict, judge_candidates

FPS = 30
SECONDS = 4.0
NOISE = 1e-3

# Data for each special case: its parameters, initial state and context (truth chosen to look like a real clip).
TRUTH = {
    "pendulum": ({"g": 9.81}, [0.8, 0.0], {"length_m": 0.6}),
    "harmonic": ({"omega": 6.0, "s0": 0.0}, [0.05, 0.0], {}),
    "damped_harmonic": ({"omega": 6.0, "s0": 0.0, "zeta": 0.05}, [0.05, 0.0], {}),
    "oscillator_dry": ({"omega": 6.0, "s0": 0.0, "f": 0.05}, [0.05, 0.0], {}),
    "free_fall": ({"g": 9.81}, [0.0, 1.0, 1.5, 2.0], {}),
    "spin_free": ({}, [0.0, 10.0], {}),
    "spin_dry": ({"c": 2.0}, [0.0, 10.0], {}),
    "spin_viscous": ({"k": 0.3}, [0.0, 10.0], {}),
    "line_constant_velocity": ({}, [0.0, 0.5], {}),
    "plane_constant_velocity": ({}, [0.0, 0.0, 0.5, 0.2], {}),
    "free_fall_3d": ({"g": 9.81}, [0.0, 1.0, 0.0, 1.5, 2.0, 0.5], {"z_scale": 1.0}),
}


def synthetic(model_id: str, seed: int = 0) -> Coordinates:
    spec = BUILTIN_MODELS[model_id]
    params, y0, ctx = TRUTH[model_id]
    t = np.arange(0, SECONDS, 1 / FPS)
    q = simulate(spec, params, np.asarray(y0, float), t, ctx)
    q = q + np.random.default_rng(seed).normal(0, NOISE, q.shape)
    return Coordinates(spec.family, t, q, tuple(f"q{i}" for i in range(q.shape[1])), dict(ctx))


@pytest.mark.parametrize(("sup", "sub"), [(sup, sub) for sup, sub, _ in NESTINGS])
def test_a_law_never_ends_worse_than_its_special_case(sup, sub):
    c = synthetic(sub)
    sub_fit, sup_fit = fit_family([BUILTIN_MODELS[sub], BUILTIN_MODELS[sup]], c)
    assert sub_fit.success and sup_fit.success
    assert sup_fit.rss <= sub_fit.rss * (1 + NESTED_REL_TOL)
    assert not sup_fit.notes


def test_every_nesting_is_declared_on_the_law():
    for sup, sub, off in NESTINGS:
        assert (sub, tuple(off.items())) in BUILTIN_MODELS[sup].nests
        assert set(off) <= set(BUILTIN_MODELS[sup].param_names) - set(BUILTIN_MODELS[sub].param_names)


def _fake(model_id: str, rss: float) -> FitResult:
    return FitResult(model_id, {}, np.zeros(2), 100, 3, rss, 0.0, 0.0, 0.0, 0.05, NOISE, success=True)


def test_the_guard_puts_a_losing_law_in_the_trail():
    specs = [BUILTIN_MODELS["damped_harmonic"], BUILTIN_MODELS["oscillator_mixed"]]
    fits = _nesting_guard(specs, {"damped_harmonic": _fake("damped_harmonic", 1.0),
                                  "oscillator_mixed": _fake("oscillator_mixed", 2.0)})
    assert fits[1].notes and "could not improve on Damped harmonic oscillator" in fits[1].notes[0]
    judged = judge_candidates(Verdict("accepted", fits[0], (fits[0],), tuple(fits), ""))
    assert "could not improve" in judged["oscillator_mixed"][1]


def test_both_frictions_from_the_truth_reach_the_noise_floor():
    # Before the corner smoothing, this fit drifted from the true parameters to three times the noise floor.
    spec = BUILTIN_MODELS["oscillator_mixed"]
    truth = {"omega": 6.0, "s0": 0.0, "zeta": 0.04, "f": 0.06}
    t = np.arange(0, 6.0, 1 / FPS)
    q = simulate(spec, truth, np.array([0.08, 0.0]), t, {})
    q = q + np.random.default_rng(1).normal(0, NOISE, q.shape)
    c = Coordinates("oscillator", t, q, ("s",), {})
    fit = fit_model(spec, c, starts=(truth,))
    assert fit.rss <= 1.05 * q.size * NOISE**2
