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


def _fake(model_id: str, k: int, bic: float, heldout: float, offset: float):
    from physicslens.physics.fit import FitResult
    sim = np.linspace(0, 1, 40)[:, None] + offset  # distinct paths, so no two count as one law written twice
    return FitResult(model_id, {}, np.zeros(2), 40, k, 1.0, bic, 0.01, heldout * 0.1, heldout, 0.01, True, sim=sim)


def test_an_extra_term_that_forecasts_worse_is_not_confirmed():
    # Damping wins on BIC (34 better) but forecasts the held-out quarter worse (13.8% against 9.9%): the gallery
    # pendulum's case. The extra term has not shown it is real, so neither law is singled out.
    v = decide([_fake("pendulum_linear_damping", 3, -1576.0, 0.138, 0.0), _fake("pendulum", 2, -1542.0, 0.099, 0.1)])
    assert v.status == "ambiguous" and "not confirmed" in v.reason
    trail = judge_candidates(v)
    assert "forecasts the held-out motion worse" in trail["pendulum_linear_damping"][1]
    assert "with fewer terms" in trail["pendulum"][1]


def test_an_extra_term_that_forecasts_as_well_still_wins():
    v = decide([_fake("pendulum_linear_damping", 3, -1576.0, 0.10, 0.0), _fake("pendulum", 2, -1542.0, 0.099, 0.1)])
    assert v.status == "accepted" and v.winner.model_id == "pendulum_linear_damping"


def test_tiny_forecast_differences_do_not_count():
    # 0.11% against 0.08%: relatively worse, but far below a percent of the motion.
    v = decide([_fake("damped_harmonic", 3, -600.0, 0.0011, 0.0), _fake("harmonic", 2, -50.0, 0.0008, 0.1)])
    assert v.status == "accepted" and v.winner.model_id == "damped_harmonic"


def test_tied_laws_report_the_parameter_they_agree_on():
    from physicslens.export.outcome import build_outcome
    from physicslens.physics.coords import to_coordinates
    from physicslens.physics.fit import fit_model
    from physicslens.physics.uncertainty import agreed_parameters

    from .test_regime import pendulum
    c = to_coordinates(pendulum([(6, 0.5, 0.0)]), "pendulum")
    laws = [(MODELS[m], fit_model(MODELS[m], c)) for m in ("pendulum", "pendulum_linear_damping")]
    agreed = agreed_parameters(laws, c, reps=12)
    g = next(a for a in agreed if a["name"] == "g")
    assert g["ci95"][0] <= 9.81 <= g["ci95"][1] and len(g["per_law"]) == 2
    result = {"status": "ambiguous", "family": "pendulum", "ranking": [], "agreed_parameters": agreed}
    o = build_outcome(result, truth={"name": "g", "value": 9.81, "source": "test"})
    assert "measured either way" in o["headline"] and o["checked_against"]["within_interval"]
    assert o["status"] == "not yet validated" and o["confidence"].startswith("g between")


def test_a_law_pinned_at_its_limits_is_not_accepted():
    # g at its lowest allowed value while drag takes its highest: it "forecasts", but only by leaning on the limits.
    pinned = _fake("linear_drag", 2, -900.0, 0.12, 0.0)
    pinned.params.update(g=MODELS["linear_drag"].params[0].lower, k=MODELS["linear_drag"].params[1].upper)
    honest = _fake("free_fall", 1, -100.0, 0.13, 0.1)
    honest.params.update(g=9.8)
    v = decide([pinned, honest])
    assert v.status == "accepted" and v.winner.model_id == "free_fall"
    assert "leaning on a limit" in judge_candidates(v)["linear_drag"][1]
    # A parameter at zero is a real answer ("no drag"), not a pinned one.
    nodrag = _fake("linear_drag", 2, -900.0, 0.12, 0.0)
    nodrag.params.update(g=9.8, k=0.0)
    assert decide([nodrag, honest]).winner.model_id == "linear_drag"


def test_a_range_spanning_decades_is_judged_on_a_log_scale():
    # A door closer's stiffness of 9 is not "at the lowest allowed value" of a range from 0.01 to 40,000; on a linear
    # scale 0.1% of that span is 40 and it was refused. A value really at the floor still is.
    from physicslens.physics.verify import at_limit
    lo, hi = MODELS["spin_spring"].params[0].lower, MODELS["spin_spring"].params[0].upper
    ok = _fake("spin_spring", 3, -1600.0, 0.01, 0.0)
    ok.params.update(kappa=9.0, phi0=0.2, k=0.8)
    assert at_limit(ok) is None
    floor = _fake("spin_spring", 3, -1600.0, 0.01, 0.0)
    floor.params.update(kappa=lo * 1.001, phi0=0.2, k=0.8)
    assert "lowest" in (at_limit(floor) or "")
    ceiling = _fake("spin_spring", 3, -1600.0, 0.01, 0.0)
    ceiling.params.update(kappa=hi * 0.999, phi0=0.2, k=0.8)
    assert "highest" in (at_limit(ceiling) or "")


def test_no_fit_says_why_in_words_a_reader_can_use():
    from physicslens.physics.verify import error_words
    # Only a pinned law forecasts: the verdict names the limit, not a forecast error that was fine.
    pinned = _fake("linear_drag", 2, -900.0, 0.12, 0.0)
    pinned.params.update(g=MODELS["linear_drag"].params[0].lower, k=MODELS["linear_drag"].params[1].upper)
    v = decide([pinned])
    assert v.status == "no_fit" and "can be trusted" in v.reason and "lowest allowed value" in v.reason
    # A diverged simulation's error (a live throw showed 1506880239378%) reads as words, not a 13-digit percentage.
    assert error_words(1.5e10) == "more than ten times the motion's own spread" and error_words(float("nan")).startswith("more")
    assert error_words(0.42) == "42%"
