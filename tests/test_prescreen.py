"""Fast honest failure: noise-like motion is refused before the simulation fits, real motion never is."""
import time

import numpy as np
import pytest
from physicslens.physics.coords import to_coordinates
from physicslens.physics.models import models_for
from physicslens.physics.prescreen import noise_like

from .test_physics import pendulum_xy, run, track

T = np.arange(0, 8, 1 / 30)
NOISE = np.random.default_rng(0).normal(0, 0.001, (len(T), 2))


def test_random_motion_is_refused_fast():
    # The same track as test_random_motion_has_no_clean_equation, which took about four minutes of fits before.
    rng = np.random.default_rng(3)
    t0 = time.monotonic()
    _, v = run(track(np.cumsum(rng.normal(0, 0.01, (240, 2)), axis=0)), "oscillator")
    assert v.status == "no_fit" and "looks like noise" in v.reason
    assert time.monotonic() - t0 < 10


@pytest.mark.parametrize("seed", range(5))
def test_random_walks_in_the_plane_are_refused(seed):
    rng = np.random.default_rng(10 + seed)
    c = to_coordinates(track(np.cumsum(rng.normal(0, 0.01, (240, 2)), axis=0)), "plane")
    assert noise_like(c, models_for("plane")) is not None


@pytest.mark.parametrize(("name", "xy", "family"), [
    ("pendulum", pendulum_xy(0.9, 0.08, 10), "pendulum"),
    ("small swing", pendulum_xy(0.12, 0.02, 6.5, length=0.5), "pendulum"),
    ("spring", np.column_stack([0.05 * np.cos(6 * T) * np.exp(-0.3 * T), 0 * T]) + NOISE, "oscillator"),
    ("constant velocity", np.column_stack([0.3 * T, 0.1 * T]) + NOISE, "plane"),
    ("free fall", np.column_stack([1.5 * T[:30], 2 * T[:30] - 4.9 * T[:30] ** 2]) + NOISE[:30], "projectile"),
    ("slow drift on a line", np.column_stack([0.02 * T + 0.01 * T**2, 0 * T]) + NOISE, "line"),
])
def test_physical_motion_is_never_refused_early(name, xy, family):
    c = to_coordinates(track(xy), family)
    assert noise_like(c, models_for(family)) is None, name
