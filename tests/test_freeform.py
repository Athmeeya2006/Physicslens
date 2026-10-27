"""Free-form laws the agent writes itself: the closed math language, and fitting them like any other model."""

import math
import subprocess
import sys

import numpy as np
import pytest
from physicslens.agent.schemas import Hypotheses
from physicslens.physics import custom
from physicslens.physics.coords import to_coordinates
from physicslens.physics.custom import FreeParam, build, register, spec_for
from physicslens.physics.expr import (
    ExprError,
    compile_expr,
    parse,
    safe_eval,
    to_latex,
    to_python,
)
from physicslens.physics.fit import fit_model
from physicslens.physics.models import MODELS
from physicslens.physics.verify import assess
from pydantic import ValidationError
from scipy.integrate import solve_ivp

from .test_physics import FPS, track

NAMES = {"s", "v", "k", "mu", "g"}


# ---------- the expression language ----------

def test_parses_and_evaluates_with_precedence():
    f = compile_expr(parse("-mu*g*sign(v) - k*v^2 + 2^3^0", NAMES))
    env = {"s": 0.0, "v": 2.0, "k": 0.5, "mu": 0.1, "g": 10.0}
    assert f(env) == pytest.approx(-0.1 * 10 * 1 - 0.5 * 4 + 2)  # 2^3^0 is right-associative: 2^(3^0) = 2


@pytest.mark.parametrize("text", [
    "__import__('os')", "v.real", "open", "lambda: 1", "v; 1", "'1'", "v[0]", "exec(v)", "sin(v, v)",
    "k*" * 80 + "k", "1e999", "", "v +", "(v", "unknown*v",
])
def test_anything_but_arithmetic_is_refused(text):
    with pytest.raises(ExprError):
        parse(text, NAMES)


def test_domain_errors_become_nan_not_crashes():
    for text in ("log(0 - 1)", "1/(v - v)", "exp(1000)", "(0 - 8)^0.5", "sqrt(0 - 1)"):
        assert math.isnan(safe_eval(compile_expr(parse(text, NAMES)), {"v": 1.0}))


def test_latex_and_python_come_from_the_same_tree():
    tree = parse("-k*v*abs(v) - mu*g*sign(v)", NAMES)
    tex = to_latex(tree, {"v": r"\dot s", "k": "k", "mu": r"\mu", "g": "g"})
    assert r"\left|\dot s\right|" in tex and r"\operatorname{sgn}" in tex and r"\mu" in tex
    src = to_python(tree)
    assert eval(src, {"np": np, "k": 0.5, "v": 2.0, "mu": 0.1, "g": 10.0}) == pytest.approx(-0.5 * 4 - 1.0)


def test_build_rejects_unusable_proposals():
    p = [FreeParam("k", "1/s", 0.0, 5.0, 1.0)]
    with pytest.raises(ExprError, match="never used"):
        build("line", "x", ["-v"], p)
    with pytest.raises(ExprError, match="not allowed"):
        build("line", "x", ["-v*v"], [FreeParam("v", "", 0.0, 1.0, 0.5)])
    with pytest.raises(ExprError, match="2 acceleration"):
        build("plane", "x", ["-k*vx"], p)
    with pytest.raises(ExprError, match="line' or 'plane"):
        build("pendulum", "x", ["-k*v"], p)
    with pytest.raises(ExprError, match="bounds"):
        build("line", "x", ["-k*v"], [FreeParam("k", "1/s", 5.0, 1.0, 2.0)])


# ---------- fitting a law the agent wrote ----------

def coasting_cart(c_drag: float = 0.8, v0: float = 2.0, seconds: float = 4.0) -> np.ndarray:
    """A cart coasting along a slightly tilted track, slowed by air drag: s'' = -c v|v|."""
    t = np.arange(0, seconds, 1 / FPS)
    s = solve_ivp(lambda _t, y: [y[1], -c_drag * y[1] * abs(y[1])], (0, seconds), [0, v0], t_eval=t,
                  rtol=1e-10, atol=1e-12).y[0]
    axis = np.array([np.cos(0.1), np.sin(0.1)])
    return np.column_stack([0.2 + s * axis[0], 0.5 + s * axis[1]])


def free(title: str, accel: list[str], params: list[FreeParam], family: str = "line"):
    return register(build(family, title, accel, params))


def test_agent_written_drag_law_beats_the_controls_and_recovers_c():
    c = to_coordinates(track(coasting_cart()), "line")
    drag = free("Coasting with air drag", ["-c*v*abs(v)"], [FreeParam("c", "1/m", 0.0, 10.0, 0.3)])
    spring = free("Spring (wrong physics)", ["-k*s"], [FreeParam("k", "1/s^2", 0.0, 100.0, 1.0)])
    ids = ["line_constant_velocity", "line_constant_accel", drag.id, spring.id]
    verdict = assess(c, [fit_model(MODELS[m], c) for m in ids])
    assert verdict.status == "accepted" and verdict.winner.model_id == drag.id
    assert verdict.winner.params["c"] == pytest.approx(0.8, rel=0.05)


def test_plane_law_with_linear_drag():
    t = np.arange(0, 1.6, 1 / FPS)
    k, g = 1.5, 9.81
    y = solve_ivp(lambda _t, s: [s[2], s[3], -k * s[2], -g - k * s[3]], (0, t[-1]), [0, 0, 2.5, 4.0], t_eval=t,
                  rtol=1e-10, atol=1e-12).y
    c = to_coordinates(track(np.column_stack([y[0], y[1]])), "plane")
    law = free("Thrown light ball with linear drag", ["-k*vx", "-g - k*vy"],
               [FreeParam("k", "1/s", 0.0, 20.0, 0.5), FreeParam("g", "m/s^2", 1.0, 30.0, 9.8)], family="plane")
    verdict = assess(c, [fit_model(MODELS[m], c) for m in ("plane_constant_velocity", "plane_constant_accel", law.id)])
    assert verdict.status == "accepted" and verdict.winner.model_id == law.id
    assert verdict.winner.params["g"] == pytest.approx(9.81, rel=0.03)


def test_hypotheses_schema_validates_free_models_and_reports_errors():
    ok = {"family": "line", "candidates": [{"model_id": "line_constant_accel", "rationale": "slows down"}],
          "free_models": [{"title": "Air drag", "acceleration": ["-c*v*abs(v)"], "rationale": "deceleration fades",
                           "params": [{"name": "c", "unit": "1/m", "lower": 0, "upper": 10, "guess": 0.3}]}]}
    assert Hypotheses.model_validate(ok).free_models[0].title == "Air drag"
    bad = {**ok, "free_models": [{**ok["free_models"][0], "acceleration": ["-c*v*abs(v) - drag"]}]}
    with pytest.raises(ValidationError, match="unknown name 'drag'"):
        Hypotheses.model_validate(bad)
    wrong_family = {**ok, "family": "pendulum", "candidates": [{"model_id": "pendulum", "rationale": "x"}]}
    with pytest.raises(ValidationError, match="only allowed with family"):
        Hypotheses.model_validate(wrong_family)


def test_stored_fit_rebuilds_its_free_model_after_a_restart():
    spec = free("Air drag", ["-c*v*abs(v)"], [FreeParam("c", "1/m", 0.0, 10.0, 0.3)])
    fit_state = {"model_id": spec.id, "free_model": custom.FREE_MODELS[spec.id].definition()}
    del MODELS[spec.id], custom.FREE_MODELS[spec.id]  # what a fresh server process looks like
    assert spec_for(fit_state).id == spec.id and spec.id in MODELS


def test_exported_script_runs_the_fitted_law(tmp_path):
    from physicslens.export.script import python_script
    from physicslens.physics.fit import simulate

    spec = free("Air drag", ["-c*v*abs(v)"], [FreeParam("c", "1/m", 0.0, 10.0, 0.3)])
    fs = {"model_id": spec.id, "params": {"c": 0.8}, "y0": [0.0, 2.0], "ctx": {}, "fps": 30.0}
    result = {"fit_state": fs, "clip": {"frames": 90, "fps": 30.0}, "status": "accepted", "parameters": []}
    (tmp_path / "sim.py").write_text(python_script(result))
    subprocess.run([sys.executable, "sim.py"], cwd=tmp_path, check=True, capture_output=True)
    out = np.loadtxt(tmp_path / "physicslens_trajectory.csv", delimiter=",", skiprows=1)
    expected = simulate(spec, fs["params"], np.array(fs["y0"]), out[:, 0], {})
    assert np.allclose(out[:, 1], expected[:, 0], atol=1e-6)


def test_known_object_size_sets_the_scale():
    from physicslens.agent.schemas import Scene
    from physicslens.perception.sam_track import ObjectTrack
    from physicslens.pipeline import RunOptions, calibrate

    from .test_pipeline import SCENE

    n = 40
    long_px = np.full(n, 90.0) + np.random.default_rng(0).normal(0, 1.0, n)
    moving = ObjectTrack("1", np.arange(n), np.zeros((n, 2)), np.zeros(n), long_px, long_px / 2, np.ones(n))
    scene = Scene.model_validate({**SCENE, "reference_kind": "none", "reference_box": None})
    cal = calibrate(scene, None, moving, RunOptions(object_size_m=1.8))
    assert cal.source == "object_size" and cal.px_per_m == pytest.approx(50.0, rel=0.01) and cal.rel_std < 0.02
