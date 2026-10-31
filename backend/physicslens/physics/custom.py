"""Free-form equations of motion: the agent writes an acceleration law, this turns it into a fittable model.

This is what lets the tool handle things outside the named families (a braking cart, a ball rolling down a
ramp, a falling leaf): the agent proposes the physics, and the same fitting and held-out verification that
judge the built-in models judge it, against the constant-velocity and constant-acceleration controls.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from .expr import (
    CONSTANTS,
    FUNCTIONS,
    NAME,
    ExprError,
    compile_expr,
    names_used,
    parse,
    safe_eval,
    to_latex,
    to_python,
)
from .models import MODELS, V_EPS, ModelSpec, Param
from .units import ACCELERATION, LENGTH, TIME, VELOCITY, Dim, check_acceleration, parse_unit

# State variables per family, in state-vector order (positions, then velocities).
STATE_VARS = {"line": ("s", "v"), "plane": ("x", "y", "vx", "vy")}
STATE_LATEX = {"s": "s", "v": r"\dot s", "x": "x", "y": "y", "vx": r"\dot x", "vy": r"\dot y"}
ACCEL_LATEX = {"line": (r"\ddot s",), "plane": (r"\ddot x", r"\ddot y")}
# More parameters need more data to pin down, and BIC (with the effective sample size, physics/rigor.py) is what
# makes each one pay for itself. An everyday clip supports 4; orbit mode (thousands of samples of a clean, rendered
# track, and a law with a central body plus a perturber) allows 6.
MAX_PARAMS = 4
MAX_PARAMS_ORBIT = 6
MAX_BOUND = 1e4
STATE_DIMS: dict[str, Dim] = {"s": LENGTH, "v": VELOCITY, "x": LENGTH, "y": LENGTH, "vx": VELOCITY, "vy": VELOCITY}
INPUT_NAME = NAME  # the same rule as parameters; checked again against every other name below
MAX_INPUT_CHARS = 8


class InputRangeError(ExprError):
    """A known input asked for outside the time it was measured over: never extrapolated."""


@dataclass(frozen=True)
class InputSignals:
    """Known, time-sampled signals a law may use but never fits: a second tracked body's position, a motor command,
    a simulated perturber's path. Times share the track's time base (seconds from its first sample)."""

    t: tuple[float, ...]
    values: dict[str, tuple[float, ...]]
    units: dict[str, str]
    provenance: dict[str, str]  # where each came from: "second tracked object", "uploaded CSV", "simulation, no pixels"

    def definition(self) -> dict[str, Any]:
        return {"t": list(self.t), "values": {k: list(v) for k, v in self.values.items()}, "units": dict(self.units),
                "provenance": dict(self.provenance)}

    @staticmethod
    def from_definition(d: dict[str, Any]) -> InputSignals:
        return InputSignals(tuple(d["t"]), {k: tuple(v) for k, v in d["values"].items()}, dict(d["units"]),
                            dict(d["provenance"]))
GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "kappa", "lambda", "mu", "nu", "xi",
         "rho", "sigma", "tau", "phi", "chi", "psi", "omega"}


@dataclass(frozen=True)
class FreeParam:
    name: str
    unit: str
    lower: float
    upper: float
    guess: float


@dataclass(frozen=True)
class FreeModel:
    """A validated proposal plus everything derived from its parse trees."""

    spec: ModelSpec
    family: str
    acceleration: tuple[str, ...]
    params: tuple[FreeParam, ...]
    python: tuple[str, ...]  # generated from the trees, for the exported script
    inputs: InputSignals | None = None
    inputs_used: tuple[str, ...] = ()
    unit_warnings: tuple[str, ...] = ()  # unit text that was not understood: those terms were not checked

    def definition(self) -> dict[str, Any]:
        """Everything needed to rebuild the model later (stored with the result, e.g. for what-ifs after a restart)."""
        out: dict[str, Any] = {"family": self.family, "title": self.spec.title, "acceleration": list(self.acceleration),
                               "params": [asdict(p) for p in self.params]}
        if self.inputs is not None and self.inputs_used:
            out["inputs"] = self.inputs.definition()
        return out


def param_latex(name: str) -> str:
    base, _, sub = name.partition("_")
    head = f"\\{base}" if base in GREEK else base
    return f"{head}_{{{sub}}}" if sub else head


def input_latex(name: str) -> str:
    """x1 -> x_{1}: a known signal reads like a second body's coordinate."""
    head = name.rstrip("0123456789")
    return f"{head}_{{{name[len(head):]}}}" if head and head != name else name


def law_latex(family: str, acceleration: list[str] | tuple[str, ...], param_names: list[str],
              input_names: list[str] | tuple[str, ...] = ()) -> str | None:
    """A written law as typeset math, for showing it before (or without) fitting it. None when it does not parse."""
    if family not in STATE_VARS or len(acceleration) != len(ACCEL_LATEX[family]):
        return None
    try:
        trees = [parse(a, set(STATE_VARS[family]) | set(param_names) | {"t"} | set(input_names)) for a in acceleration]
    except ExprError:
        return None
    symbols = {**STATE_LATEX, "t": "t", **{n: input_latex(n) for n in input_names}, **{n: param_latex(n) for n in param_names}}
    return r",\quad ".join(f"{lhs} = {to_latex(t, symbols)}" for lhs, t in zip(ACCEL_LATEX[family], trees, strict=True))


def build(family: str, title: str, acceleration: list[str], params: list[FreeParam],
          inputs: InputSignals | None = None, max_params: int = MAX_PARAMS) -> FreeModel:
    """Validate and compile a proposal. Raises ExprError with a message the agent can act on.

    The law may use the state, its parameters, the time t (seconds from the track's first sample) and the names of
    any known inputs. Its units are checked before anything is fitted."""
    if family not in STATE_VARS:
        raise ExprError(f"free-form models are written for family 'line' or 'plane', not {family!r}")
    state = STATE_VARS[family]
    want = len(state) // 2
    if len(acceleration) != want:
        raise ExprError(f"a {family} model needs {want} acceleration expression(s) ({', '.join(ACCEL_LATEX[family])}), got {len(acceleration)}")
    if len(params) > max_params:
        raise ExprError(f"at most {max_params} parameters")
    # Also reserved: names the exported Python script uses for itself.
    reserved = set(state) | set(FUNCTIONS) | set(CONSTANTS) | {"t", "np", "p", "state", "L", "dv", "ax", "ay"}
    names: list[str] = []
    for p in params:
        if not NAME.match(p.name) or p.name in reserved:
            raise ExprError(f"parameter name {p.name!r} is not allowed (letters, digits, underscore; not a state variable or function)")
        if p.name in names:
            raise ExprError(f"parameter {p.name!r} is declared twice")
        if not (math.isfinite(p.lower) and math.isfinite(p.upper) and -MAX_BOUND <= p.lower < p.upper <= MAX_BOUND):
            raise ExprError(f"parameter {p.name!r} needs finite bounds with lower < upper within +-{MAX_BOUND:g}")
        if inputs is not None and p.name in inputs.values:
            raise ExprError(f"{p.name!r} is a known input (measured, never fitted); it cannot also be a parameter")
        names.append(p.name)
    input_names = sorted(inputs.values) if inputs is not None else []
    for name in input_names:
        if not INPUT_NAME.match(name) or len(name) > MAX_INPUT_CHARS or name in reserved or name in FUNCTIONS:
            raise ExprError(f"input name {name!r} is not allowed (letters and digits, at most {MAX_INPUT_CHARS} "
                            "characters; not a state variable, parameter or function)")
    trees = [parse(a, set(state) | set(names) | {"t"} | set(input_names)) for a in acceleration]
    used = set().union(*(names_used(t) for t in trees))
    unused = set(names) - used
    if unused:
        raise ExprError(f"parameters {sorted(unused)} are declared but never used")
    inputs_used = tuple(n for n in input_names if n in used)
    time_dependent = "t" in used or bool(inputs_used)

    # Units before anything is fitted: an inconsistent law is rejected with a message the agent can act on.
    dims: dict[str, Dim | None] = {**{n: STATE_DIMS[n] for n in state}, "t": TIME,
                                   **{p.name: parse_unit(p.unit) for p in params}}
    for name in inputs_used:
        dims[name] = parse_unit(inputs.units.get(name)) if inputs is not None else None
    report = check_acceleration(trees, dims)
    if not report.ok:
        raise ExprError(f"units: {report.message} (an acceleration is {ACCELERATION})")
    warnings = tuple(f"the unit of {n!r} was not understood, so terms with it were not checked for units"
                     for n in report.unknown if n in used)

    fns = [compile_expr(t) for t in trees]
    d = len(state) // 2
    splines: dict[str, CubicSpline] = {}
    t_lo = t_hi = 0.0
    if inputs is not None and inputs_used:
        t_in = np.asarray(inputs.t, dtype=float)
        if len(t_in) < 4 or np.any(np.diff(t_in) <= 0):
            raise ExprError("known inputs need at least 4 samples at increasing times")
        splines = {n: CubicSpline(t_in, np.asarray(inputs.values[n], dtype=float)) for n in inputs_used}
        # Half a sample of slack: the solver evaluates a hair past the last output time.
        slack = 0.5 * float(np.median(np.diff(t_in)))
        t_lo, t_hi = float(t_in[0]) - slack, float(t_in[-1]) + slack

    def env_at(s: np.ndarray, p: dict[str, float], ctx: dict[str, float], t: float | None) -> dict[str, float]:
        env = dict(p)
        env.update(zip(state, (float(v) for v in s), strict=True))
        scale = ctx.get("v_eps_scale", 1.0)
        if scale > 1.0:  # the fit's early windows: sign() gets the library's smoothed corner (expr.compile_expr)
            env["_sign_eps"] = V_EPS * scale
        if t is not None:
            env["t"] = float(t)
            if splines and not t_lo <= t <= t_hi:
                raise InputRangeError(f"known inputs cover {t_lo:.3g} s to {t_hi:.3g} s; asked for t = {t:.3g} s")
            for n, sp in splines.items():
                env[n] = float(sp(t))
        return env

    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float], t: float | None = None) -> np.ndarray:
        env = env_at(s, p, ctx, t)
        # Always numbers: a term that evaluates to true/false would make the fitter subtract booleans and crash.
        return np.array([*s[d:], *(safe_eval(f, env) for f in fns)], dtype=float)

    symbols = {**STATE_LATEX, "t": "t", **{n: input_latex(n) for n in inputs_used},
               **{n: param_latex(n) for n in names}}
    latex = r",\quad ".join(f"{lhs} = {to_latex(t, symbols)}" for lhs, t in zip(ACCEL_LATEX[family], trees, strict=True))
    key = json.dumps({"family": family, "acceleration": acceleration,
                      "params": [[p.name, p.lower, p.upper] for p in params],
                      # Two runs with different measured inputs are different models.
                      "inputs": hashlib.sha256(json.dumps(inputs.definition() if inputs and inputs_used else None)
                                               .encode()).hexdigest()[:12]}, sort_keys=True)
    model_id = "custom_" + hashlib.sha256(key.encode()).hexdigest()[:10]
    spec = ModelSpec(model_id, family, title.strip()[:60] or "Free-form law", latex,
                     tuple(Param(p.name, param_latex(p.name), p.unit, p.lower, p.upper,
                                 float(min(max(p.guess, p.lower), p.upper))) for p in params), rhs,
                     time_dependent=time_dependent)
    return FreeModel(spec, family, tuple(acceleration), tuple(params), tuple(to_python(t) for t in trees),
                     inputs if inputs_used else None, inputs_used, warnings)


FREE_MODELS: dict[str, FreeModel] = {}


def register(model: FreeModel) -> ModelSpec:
    """Ids are content hashes, so registering the same proposal twice is harmless."""
    FREE_MODELS[model.spec.id] = model
    MODELS[model.spec.id] = model.spec
    return model.spec


def restore(definition: dict[str, Any]) -> ModelSpec:
    inputs = InputSignals.from_definition(definition["inputs"]) if definition.get("inputs") else None
    return register(build(definition["family"], definition["title"], list(definition["acceleration"]),
                          [FreeParam(**p) for p in definition["params"]], inputs, max_params=MAX_PARAMS_ORBIT))


def spec_for(fit_state: dict[str, Any]) -> ModelSpec:
    """The model a stored fit refers to, rebuilding a free-form one from its saved definition if needed."""
    model_id = fit_state["model_id"]
    if model_id in MODELS:
        return MODELS[model_id]
    if fit_state.get("free_model"):
        return restore(fit_state["free_model"])
    raise KeyError(f"Unknown model {model_id!r}")
