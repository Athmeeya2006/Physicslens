"""Predictions from a law found in video, with the planets added as known forcing (sky/prediction.py)."""
import json
from pathlib import Path

import numpy as np
from physicslens.sky.ephem import J2000, catalogue_system, sun_gm
from physicslens.sky.nbody import simulate
from physicslens.sky.prediction import PROVENANCE, law_extra, lift, planet_paths, state_covariance

ROOT = Path(__file__).resolve().parents[1]


def gallery(name: str) -> tuple[dict, dict]:
    result = json.loads((ROOT / "gallery" / f"gallery-{name}" / "result.json").read_text())
    folder = {"halley": "halley", "apophis": "orbit"}[name]
    return result, json.loads((ROOT / "data" / folder / "render.json").read_text())


def test_the_fitted_start_is_lifted_back_into_the_videos_plane():
    result, meta = gallery("halley")
    jd0, state = lift(result["fit_state"], meta)
    basis = np.asarray(meta["basis"])
    assert jd0 == meta["start_jd"]
    np.testing.assert_allclose(basis[:2] @ state[:3], result["fit_state"]["y0"][:2], atol=1e-12)
    assert abs(basis[2] @ state[:3]) < 1e-12  # nothing out of the plane: the video cannot see it


def test_the_apophis_laws_own_extra_term_is_kept():
    result, meta = gallery("apophis")
    extra = law_extra(result["fit_state"], meta)
    v = np.array([[0.0, 0.017, 0.0]])
    np.testing.assert_allclose(extra(np.zeros((1, 3)), v), -result["fit_state"]["params"]["b"] * v)
    halley, hmeta = gallery("halley")
    assert law_extra(halley["fit_state"], hmeta) is None


def test_the_fits_covariance_is_a_covariance():
    result, _ = gallery("halley")
    mean, cov = state_covariance(result)
    assert mean[0] == result["fit_state"]["params"]["mu"]
    np.testing.assert_allclose(cov, cov.T, rtol=1e-6, atol=1e-30)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_planet_paths_follow_the_planets_own_integration():
    paths = planet_paths(J2000, J2000 + 400)
    jupiter = next(p for p in paths if p.name == "Jupiter")
    direct = simulate(sun_gm(), catalogue_system(["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn",
                                                  "Uranus", "Neptune", "Pluto", "Ceres"]), J2000, np.array([J2000 + 250.5]))
    np.testing.assert_allclose(jupiter.where(J2000 + 250.5), direct.of("Jupiter")[0][0], atol=1e-9)
    assert "not discovered" in PROVENANCE
