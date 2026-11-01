"""Time and known inputs in the law language, and the units check that runs before any fit."""
import json
import subprocess
import sys

import numpy as np
import pytest
from physicslens.agent.schemas import LAW_CONTEXT, FreeModelIn, Hypotheses, LawContext
from physicslens.export.script import python_script
from physicslens.inputs import InputError, parse_inputs_csv, track_signals
from physicslens.physics.coords import Coordinates
from physicslens.physics.custom import (
    MAX_PARAMS_ORBIT,
    FreeParam,
    InputRangeError,
    InputSignals,
    build,
    register,
    restore,
)
from physicslens.physics.expr import ExprError
from physicslens.physics.fit import fit_model, simulate

FPS = 30
T = np.arange(0, 8, 1 / FPS)


def p(name, unit, lo, hi, guess):
    return FreeParam(name, unit, lo, hi, guess)


DRIVEN = ["-w^2*s - 2*z*w*v + A*cos(W*t)"]
DRIVEN_PARAMS = [p("w", "1/s", 0.5, 20, 3.2), p("z", "", 0, 1, 0.05), p("A", "m/s^2", 0, 5, 0.3), p("W", "1/s", 0.5, 20, 2.1)]


def test_time_is_a_name_a_law_may_use_and_shows_in_latex_and_python():
    m = build("line", "Driven spring", DRIVEN, DRIVEN_PARAMS)
    assert m.spec.time_dependent and "t\\right)" in m.spec.latex and "np.cos((W * t))" in m.python[0]
    with pytest.raises(ExprError, match="not allowed"):
        build("line", "t as a parameter", ["-t*s"], [p("t", "1/s^2", 0, 1, 0.5)])


def test_a_driven_oscillator_is_recovered():
    truth = {"w": 3.0, "z": 0.04, "A": 0.25, "W": 2.2}
    spec = build("line", "Driven spring", DRIVEN, DRIVEN_PARAMS).spec
    q = simulate(spec, truth, np.array([0.05, 0.0]), T, {})
    q = q + np.random.default_rng(0).normal(0, 2e-4, q.shape)
    fit = fit_model(spec, Coordinates("line", T, q, ("s",), {}))
    assert fit.success
    assert fit.params["A"] == pytest.approx(truth["A"], rel=0.02) and fit.params["W"] == pytest.approx(truth["W"], rel=0.02)


def attractor_inputs() -> InputSignals:
    x1, y1 = 0.3 * np.cos(0.6 * T), 0.2 * np.sin(0.6 * T)
    return InputSignals(tuple(T), {"x1": tuple(x1), "y1": tuple(y1)}, {"x1": "m", "y1": "m"},
                        {"x1": "simulation, no pixels", "y1": "simulation, no pixels"})


def test_a_law_using_a_moving_attractors_measured_path_is_recovered():
    law = build("plane", "Pulled toward a moving point", ["-k*(x - x1) - c*vx", "-k*(y - y1) - c*vy"],
                [p("k", "1/s^2", 0.1, 50, 3.0), p("c", "1/s", 0, 5, 0.2)], attractor_inputs())
    assert law.inputs_used == ("x1", "y1") and "x_{1}" in law.spec.latex
    q = simulate(law.spec, {"k": 6.0, "c": 0.3}, np.array([0.5, 0.0, 0.0, 0.0]), T, {})
    fit = fit_model(law.spec, Coordinates("plane", T, q + np.random.default_rng(1).normal(0, 1e-3, q.shape),
                                          ("x", "y"), {}))
    assert fit.params["k"] == pytest.approx(6.0, rel=0.02) and fit.params["c"] == pytest.approx(0.3, rel=0.1)


def test_inputs_are_never_parameters_and_never_extrapolated():
    with pytest.raises(ExprError, match="known input"):
        build("plane", "x1 fitted", ["-x1*x", "-x1*y"], [p("x1", "1/s^2", 0, 5, 1)], attractor_inputs())
    law = build("plane", "Pulled", ["-k*(x - x1)", "-k*(y - y1)"], [p("k", "1/s^2", 0.1, 50, 3.0)], attractor_inputs())
    with pytest.raises(InputRangeError):
        law.spec.rhs(np.zeros(4), {"k": 1.0}, {}, 99.0)


def test_a_law_with_inputs_survives_a_restart():
    law = build("plane", "Pulled", ["-k*(x - x1)", "-k*(y - y1)"], [p("k", "1/s^2", 0.1, 50, 3.0)], attractor_inputs())
    again = restore(law.definition())
    assert again.id == law.spec.id and again.time_dependent


def test_the_exported_script_reproduces_the_fit(tmp_path):
    law = build("plane", "Pulled", ["-k*(x - x1) - c*vx", "-k*(y - y1) - c*vy"],
                [p("k", "1/s^2", 0.1, 50, 3.0), p("c", "1/s", 0, 5, 0.2)], attractor_inputs())
    spec = register(law)
    params, y0 = {"k": 6.0, "c": 0.3}, [0.5, 0.0, 0.0, 0.0]
    seconds = 6.0
    expected = simulate(spec, params, np.array(y0), np.arange(int(seconds * FPS)) / FPS, {})
    result = {"status": "accepted", "parameters": [], "clip": {"frames": int(seconds * FPS), "fps": FPS},
              "fit_state": {"model_id": spec.id, "params": params, "y0": y0, "ctx": {}, "fps": FPS,
                            "free_model": law.definition()}}
    script = tmp_path / "sim.py"
    script.write_text(python_script(result, seconds))
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True, capture_output=True)
    got = np.loadtxt(tmp_path / "physicslens_trajectory.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(got[:, 1:3], expected, atol=1e-6)


# Library laws written in the agent's language, with their library units: every one must pass the units check.
LIBRARY_EQUIVALENTS = [
    ("line", ["-mu*9.81*sign(v)"], [p("mu", "", 0, 2, 0.3)]),
    ("line", ["-k*v - f*sign(v)"], [p("k", "1/s", 0, 5, 1), p("f", "m/s^2", 0, 5, 1)]),
    ("line", ["-w^2*s - 2*z*w*v"], [p("w", "rad/s", 0.1, 20, 3), p("z", "", 0, 2, 0.1)]),
    ("line", ["-c*v*abs(v)"], [p("c", "1/m", 0, 5, 1)]),
    ("line", ["-w^2*s - b*s^3"], [p("w", "1/s", 0.1, 20, 3), p("b", "1/(m^2 s^2)", -100, 100, 0)]),
    ("line", ["a0"], [p("a0", "m/s^2", -10, 10, 0)]),
    ("plane", ["-k*vx", "-g - k*vy"], [p("g", "m/s^2", 0, 20, 9.8), p("k", "1/s", 0, 5, 0.1)]),
    ("plane", ["-c*hypot(vx,vy)*vx", "-g - c*hypot(vx,vy)*vy"], [p("g", "m/s^2", 0, 20, 9.8), p("c", "1/m", 0, 5, 0.1)]),
    ("plane", ["-mu*x/(x^2+y^2)^1.5", "-mu*y/(x^2+y^2)^1.5"], [p("mu", "AU^3/day^2", 0, 1, 3e-4)]),
    ("plane", ["-mu*x/hypot(x,y)^3", "-mu*y/hypot(x,y)^3"], [p("mu", "m^3/s^2", 0, 1, 0.1)]),
]


@pytest.mark.parametrize(("family", "acc", "params"), LIBRARY_EQUIVALENTS)
def test_every_library_equivalent_law_passes_the_units_check(family, acc, params):
    build(family, "ok", acc, params)


WRONG = [
    ("line", ["-k*s"], [p("k", "1/s", 0, 5, 1)], "L T^-1"),
    ("line", ["-k*v"], [p("k", "", 0, 5, 1)], "L T^-1"),
    ("line", ["f*sin(s)"], [p("f", "m/s^2", 0, 5, 1)], "sin()"),
    ("line", ["f*exp(t)"], [p("f", "m/s^2", 0, 5, 1)], "exp()"),
    ("line", ["-k*v + s"], [p("k", "1/s", 0, 5, 1)], "term 2 has units L,"),
    ("line", ["-k*(v + s)"], [p("k", "1/s", 0, 5, 1)], "adds"),
    ("line", ["-g"], [p("g", "m/s", 0, 20, 9)], "L T^-1"),
    ("line", ["s^k"], [p("k", "1/s", 0, 5, 1)], "power with units"),
    ("line", ["-w^2*s"], [p("w", "1/m", 0, 5, 1)], "L^-1"),
    ("line", ["-2*k*s"], [p("k", "1/s", 0, 5, 1)], "L T^-1"),
    ("plane", ["-mu*x/(x^2+y^2)", "-mu*y/(x^2+y^2)"], [p("mu", "m^3/s^2", 0, 1, 0.1)], "L^2"),
    ("plane", ["-k*hypot(x, vx)", "0"], [p("k", "1/s", 0, 5, 1)], "hypot"),
]


@pytest.mark.parametrize(("family", "acc", "params", "needle"), WRONG)
def test_dimensionally_wrong_laws_are_rejected_before_fitting(family, acc, params, needle):
    with pytest.raises(ExprError, match="units") as e:
        build(family, "wrong", acc, params)
    assert needle in str(e.value)


def test_a_fitted_exponent_leaves_its_term_unchecked_not_rejected():
    # The free power law that measured Kepler's exponent: mu's unit then depends on n, so the term is not checked.
    m = build("plane", "free power", ["-mu*x/(x^2+y^2)^(n/2)", "-mu*y/(x^2+y^2)^(n/2)"],
              [p("mu", "AU^3/day^2", 0, 1, 3e-4), p("n", "", 2, 4, 3)], max_params=MAX_PARAMS_ORBIT)
    assert m.spec.params[1].name == "n"


def test_unknown_unit_text_warns_instead_of_rejecting():
    m = build("line", "odd units", ["-k*v"], [p("k", "furlongs per fortnight", 0, 5, 1)])
    assert m.unit_warnings and "not understood" in m.unit_warnings[0]


def test_orbit_mode_allows_six_parameters():
    six = [p(n, "m/s^2", 0, 1, 0.5) for n in ("a", "b", "c", "d", "e", "f")]
    acc = ["a + b + c", "d + e + f"]
    with pytest.raises(ExprError, match="at most 4"):
        build("plane", "six", acc, six)
    build("plane", "six", acc, six, max_params=MAX_PARAMS_ORBIT)


def test_the_schema_checks_proposals_against_the_runs_inputs():
    token = LAW_CONTEXT.set(LawContext(signals=attractor_inputs()))
    try:
        law = {"title": "Pulled", "acceleration": ["-k*(x - x1)", "-k*(y - y1)"], "rationale": "r",
               "params": [{"name": "k", "unit": "1/s^2", "lower": 0.1, "upper": 50, "guess": 3}],
               "inputs_used": ["x1", "y1"]}
        Hypotheses.model_validate({"family": "plane", "candidates": [{"model_id": "plane_constant_velocity",
                                                                      "rationale": "control"}],
                                   "free_models": [law]})
        # A law naming an input the run does not have is set aside with the reason; the rest of the answer stands.
        h = Hypotheses.model_validate({"family": "plane", "candidates": [{"model_id": "plane_constant_velocity",
                                                                          "rationale": "control"}],
                                       "free_models": [{**law, "inputs_used": ["z9"]}, law]})
        assert [f.title for f in h.free_models] == ["Pulled"] and "no known input" in h.set_aside[0]
    finally:
        LAW_CONTEXT.reset(token)
    assert FreeModelIn.model_fields["inputs_used"].default_factory() == []


def test_inputs_csv_is_parsed_and_shifted_to_the_track():
    raw = parse_inputs_csv(b"t,x1[m],u\n0,0,1\n0.1,0.1,1\n0.2,0.2,1\n0.3,0.3,1\n0.4,0.4,1\n")
    assert raw["units"] == {"x1": "m", "u": ""} and raw["provenance"]["x1"] == "uploaded CSV"
    sig = track_signals(raw, first_frame=3, fps=30)
    assert sig.t[0] == pytest.approx(-0.1)
    for bad, why in [(b"time,x\n", "header and at least 4"), (b"x,t\n0,0\n1,1\n2,2\n3,3\n", "first column"),
                     (b"t,x\n0,0\n0,1\n2,2\n3,3\n", "increase"), (b"t,x\n0,0\n1,nan\n2,2\n3,3\n", "NaN"),
                     (b"t,bad name\n0,0\n1,1\n2,2\n3,3\n", "letters and digits")]:
        with pytest.raises(InputError, match=why):
            parse_inputs_csv(bad)
    assert json.dumps(raw)  # plain JSON: stored with the run


def test_upload_rejects_a_bad_inputs_file_before_keeping_the_video():
    from fastapi.testclient import TestClient
    from physicslens import app as app_module
    client = TestClient(app_module.app)
    r = client.post("/api/runs", files={"video": ("a.mp4", b"x", "video/mp4"),
                                        "inputs_csv": ("in.csv", b"x,t\n0,0\n1,1\n2,2\n3,3\n", "text/csv")})
    assert r.status_code == 400 and "first column" in r.json()["detail"]


def test_a_known_movers_track_becomes_inputs_in_the_subjects_frame():
    from physicslens.inputs import merge_signals, mover_signals
    frames = np.arange(10, 40)
    px = np.column_stack([100 + 2.0 * (frames - 10), np.full(30, 200.0)])
    sig = mover_signals(frames, px, first_frame=12, fps=30, px_per_m=100, origin_px=np.array([100.0, 200.0]), name="cart")
    assert sig.t[0] == pytest.approx(-2 / 30) and sig.values["x2"][1] == pytest.approx(0.02)
    assert sig.provenance["x2"] == "second tracked object (cart)" and sig.units["y2"] == "m"
    both = merge_signals(attractor_inputs(), sig, np.arange(0, 0.9, 1 / 30))
    assert set(both.values) == {"x1", "y1", "x2", "y2"} and both.t[0] >= 0


def test_a_perturber_law_without_the_indirect_term_is_flagged():
    sig = attractor_inputs()
    kw = [p("mu", "m^3/s^2", 0, 10, 1), p("gk", "m^3/s^2", 0, 1, 0.1)]
    direct_only = build("plane", "direct only", ["-mu*x/hypot(x,y)^3 + gk*(x1 - x)/hypot(x1 - x, y1 - y)^3",
                                                 "-mu*y/hypot(x,y)^3 + gk*(y1 - y)/hypot(x1 - x, y1 - y)^3"], kw, sig)
    assert direct_only.indirect_warning and "indirect term" in direct_only.indirect_warning
    full = build("plane", "with the indirect term",
                 ["-mu*x/hypot(x,y)^3 + gk*((x1 - x)/hypot(x1 - x, y1 - y)^3 - x1/hypot(x1, y1)^3)",
                  "-mu*y/hypot(x,y)^3 + gk*((y1 - y)/hypot(x1 - x, y1 - y)^3 - y1/hypot(x1, y1)^3)"], kw, sig)
    assert full.indirect_warning is None
