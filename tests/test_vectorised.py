"""Batch evaluation of library laws gives exactly the state-by-state numbers (models.accelerations)."""
import numpy as np
import pytest
from physicslens.physics.models import BUILTIN_MODELS, accelerations, derivative

CTX = {"length_m": 0.6, "z_scale": 1.0, "L1": 0.3, "L2": 0.3}


@pytest.mark.parametrize("model_id", sorted(m for m in BUILTIN_MODELS if m != "double_pendulum"))
def test_batch_equals_one_by_one(model_id):
    spec = BUILTIN_MODELS[model_id]
    rng = np.random.default_rng(0)
    d = {"pendulum": 1, "oscillator": 1, "rotation": 1, "line": 1, "plane": 2, "projectile": 2, "projectile3d": 3}[spec.family]
    states = rng.normal(0, 1, (50, 2 * d))
    params = {p.name: (p.guess if p.guess is not None else (p.lower + p.upper) / 2) for p in spec.params}
    times = np.linspace(0, 1, 50)
    batch = accelerations(spec, states, params, CTX, times)
    one = np.array([derivative(spec, s, params, CTX, t)[d:] for s, t in zip(states, times, strict=True)])
    np.testing.assert_allclose(batch, one, rtol=1e-12, atol=1e-12)
