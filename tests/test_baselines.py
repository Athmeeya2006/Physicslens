"""The classical baselines are fair: on clean synthetic motion each recovers its own law's parameters."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from physicslens.physics.coords import Coordinates

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("baselines", ROOT / "scripts" / "baselines.py")
baselines = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baselines)


def test_kepler_curve_fit_recovers_gm_and_forecasts():
    gm, a, e = 2.959e-4, 1.1, 0.19
    t = np.linspace(0, 900, 400)
    q = baselines._kepler_xy(t, gm, a, e, 0.7, 0.3, 1.0)
    c = Coordinates("plane", t, q, ("x", "y"), {})
    pred, fitted, _ = baselines.textbook_kepler(c, 300)
    assert fitted == pytest.approx(gm, rel=1e-4)
    assert baselines.nrmse(pred, q, 300) < 1e-4


def test_a_clockwise_eccentric_orbit_too():
    t = np.linspace(0, 400, 500)
    q = baselines._kepler_xy(t, 1.0, 3.0, 0.6, -1.2, 2.0, -1.0)
    _, fitted, _ = baselines.textbook_kepler(Coordinates("plane", t, q, ("x", "y"), {}), 375)
    assert fitted == pytest.approx(1.0, rel=1e-3)


def test_harmonic_start_finds_the_swing_on_a_short_window():
    t = np.arange(0, 4.9, 1 / 25)
    x = 0.12 * np.cos(4.43 * t + 0.4) + np.random.default_rng(0).normal(0, 0.003, len(t))
    assert baselines._first_frequency(t, x) == pytest.approx(4.43, rel=0.02)
    c = Coordinates("pendulum", t, x[:, None], ("theta",), {})
    _, w, _ = baselines.textbook_oscillation(c, len(t), damped=False)
    assert w == pytest.approx(4.43, rel=0.005)


def test_divergence_is_reported_not_raised():
    assert baselines.nrmse(None, np.zeros((4, 1)), 2) == math.inf
    assert baselines._cell("k", None, None, math.inf) == "k = n/a; diverged"
