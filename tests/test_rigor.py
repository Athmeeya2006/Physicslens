"""Rolling-origin forecasts and BIC with the effective sample size (physics/rigor.py, verify.strengthen)."""
import numpy as np
from physicslens.physics.coords import Coordinates
from physicslens.physics.fit import fit_family, simulate
from physicslens.physics.models import MODELS
from physicslens.physics.rigor import ROLLING_ORIGINS, effective_n, rolling_origin
from physicslens.physics.verify import assess, decide, judge_candidates

T = np.arange(0, 8, 1 / 30)


def oscillator(zeta: float, noise: np.ndarray) -> Coordinates:
    q = simulate(MODELS["damped_harmonic"], {"omega": 4.0, "s0": 0.0, "zeta": zeta}, np.array([0.1, 0.0]), T, {})
    return Coordinates("oscillator", T, q + noise[:, None], ("s",), {})


def correlated_noise(rho: float = 0.95, sigma: float = 0.0015, seed: int = 0) -> np.ndarray:
    w = np.random.default_rng(seed).normal(0, sigma, len(T))
    e = np.zeros(len(T))
    for i in range(1, len(T)):
        e[i] = rho * e[i - 1] + w[i]
    return e


def test_correlated_residuals_do_not_make_a_tiny_term_decisive():
    # A barely damped spring under smoothly wandering tracking error: counting every sample as independent evidence
    # calls the damping term decisive; the effective sample size says the data can't tell.
    c = oscillator(0.004, correlated_noise())
    fits = fit_family([MODELS["harmonic"], MODELS["damped_harmonic"]], c)
    raw = decide(fits)
    assert raw.status == "accepted" and raw.winner.model_id == "damped_harmonic"
    fair = assess(c, fits)
    assert fair.status == "ambiguous" and fair.n_eff < 0.1 * c.q.size
    assert all(f.bic_raw is not None for f in fair.ranked)


def test_white_noise_keeps_nearly_every_sample():
    n_eff, rho = effective_n(np.random.default_rng(1).normal(0, 1, (300, 2)))
    assert abs(rho) < 0.1 and n_eff > 0.8 * 600


def test_the_right_law_forecasts_from_every_origin_and_a_wrong_one_does_not():
    c = oscillator(0.05, np.random.default_rng(2).normal(0, 0.001, len(T)))
    good, = fit_family([MODELS["damped_harmonic"]], c)
    rolling = rolling_origin(MODELS["damped_harmonic"], c, good)
    assert len(rolling) == len(ROLLING_ORIGINS) and max(r["nrmse"] for r in rolling) < 0.05
    wrong = rolling_origin(MODELS["damped_harmonic"], c, type(good)(**{**good.__dict__, "params": {**good.params, "omega": 4.4}}))
    assert np.median([r["nrmse"] for r in wrong]) > 0.15


def test_the_verdict_carries_the_rolling_forecasts_and_says_why_a_law_failed_them():
    c = oscillator(0.05, np.random.default_rng(3).normal(0, 0.001, len(T)))
    v = assess(c, fit_family([MODELS["harmonic"], MODELS["damped_harmonic"]], c))
    assert v.status == "accepted" and len(v.winner.rolling) == len(ROLLING_ORIGINS)
    assert v.winner.model_id in judge_candidates(v)
    live = assess(c, fit_family([MODELS["damped_harmonic"]], c), rolling=False)
    assert live.winner.rolling == ()
