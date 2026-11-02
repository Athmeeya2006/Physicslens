"""Perturbation discovery: structured residuals, period hints, and no planet invented from noise."""
import math

import numpy as np
import pytest
from physicslens.physics.residuals import FAP_SIGNIFICANT, period_matches, spectrum
from physicslens.sky import scenarios as S
from physicslens.sky.discover import discover


def circle(t: np.ndarray, period: float = 365.25, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    w = 2 * math.pi / period
    pos = radius * np.column_stack([np.cos(w * t), np.sin(w * t)])
    vel = radius * w * np.column_stack([-np.sin(w * t), np.cos(w * t)])
    return pos, vel


def test_a_planted_periodic_residual_is_found_and_white_noise_is_not():
    t = np.arange(0, 2000, 2.0)
    pos, vel = circle(t)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1e-4, pos.shape)
    quiet = spectrum(t, pos, vel, noise)
    assert not quiet.significant
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    planted = noise + 5e-4 * np.sin(2 * math.pi * t / 400.0)[:, None] * radial
    loud = spectrum(t, pos, vel, planted)
    assert loud.significant and loud.peaks[0].component == "radial"
    assert loud.peaks[0].period == pytest.approx(400.0, rel=0.05) and loud.peaks[0].fap < FAP_SIGNIFICANT


def test_period_matching_names_the_synodic_period():
    t = np.arange(0, 22000, 10.0)  # longer than the synodic period, or the periodogram cannot resolve it
    pos, vel = circle(t, period=4332.6)  # Jupiter-like target
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    synodic = 1 / abs(1 / 4332.6 - 1 / 10759.2)  # with a Saturn-like body
    resid = 1e-3 * np.sin(2 * math.pi * t / synodic)[:, None] * radial + np.random.default_rng(1).normal(0, 1e-5, pos.shape)
    matches = period_matches(spectrum(t, pos, vel, resid).peaks, 4332.6, {"Saturn": 10759.2, "Mars": 687.0})
    assert matches and matches[0]["body"] == "Saturn" and matches[0]["kind"] == "synodic period"


def test_no_planet_is_invented_from_noise_quick():
    # The fast CI version of the check below: two fixed seeds (deterministic, so never flaky), both must stay clean.
    # The full 20-seed bound runs in the slow job and nightly.
    sr = S.run(S.load("apophis_sun_only"))
    for s in (0, 1):
        assert "No perturbation needed" in (discover(sr, noise_px=2.0, seed=s, bootstrap_reps=8).get("perturbation") or "")


@pytest.mark.slow
def test_no_planet_is_invented_from_noise():
    # A pure Sun-only orbit plus tracker noise: the system must not find a perturbation (at least 19 of 20 seeds).
    sr = S.run(S.load("apophis_sun_only"))
    flagged = sum("No perturbation needed" not in (discover(sr, noise_px=2.0, seed=s, bootstrap_reps=8)
                                                   .get("perturbation") or "") for s in range(20))
    assert flagged <= 1


@pytest.mark.slow
def test_le_verrier_says_so_when_neptune_cannot_be_seen():
    # At a video tracker's noise Neptune moves Uranus by a third of a pixel: the system must say the data cannot show
    # a perturbation, not invent a planet. (The numbers at lower noise are recorded in data/sky/leverrier.json.)
    sr = S.run(S.load("uranus_neptune_hidden"))
    assert not sr.detectability.visible
    r = discover(sr, noise_px=2.0, bootstrap_reps=8)
    assert "hidden body" not in (r.get("law") or {}).get("title", "")
    assert "cannot" in r["perturbation"] or "No perturbation needed" in r["perturbation"]
