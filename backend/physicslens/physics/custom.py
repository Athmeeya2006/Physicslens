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
from .models import MODELS, ModelSpec, Param

# State variables per family, in state-vector order (positions, then velocities).
STATE_VARS = {"line": ("s", "v"), "plane": ("x", "y", "vx", "vy")}
STATE_LATEX = {"s": "s", "v": r"\dot s", "x": "x", "y": "y", "vx": r"\dot x", "vy": r"\dot y"}
ACCEL_LATEX = {"line": (r"\ddot s",), "plane": (r"\ddot x", r"\ddot y")}
MAX_PARAMS = 4
MAX_BOUND = 1e4
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

    def definition(self) -> dict[str, Any]:
        """Everything needed to rebuild the model later (stored with the result, e.g. for what-ifs after a restart)."""
        return {"family": self.family, "title": self.spec.title, "acceleration": list(self.acceleration),
                "params": [asdict(p) for p in self.params]}


def param_latex(name: str) -> str:
    base, _, sub = name.partition("_")
    head = f"\\{base}" if base in GREEK else base
    return f"{head}_{{{sub}}}" if sub else head


def law_latex(family: str, acceleration: list[str] | tuple[str, ...], param_names: list[str]) -> str | None:
    """A written law as typeset math, for showing it before (or without) fitting it. None when it does not parse."""
    if family not in STATE_VARS or len(acceleration) != len(ACCEL_LATEX[family]):
        return None
    try:
        trees = [parse(a, set(STATE_VARS[family]) | set(param_names)) for a in acceleration]
    except ExprError:
        return None
    symbols = {**STATE_LATEX, **{n: param_latex(n) for n in param_names}}
    return r",\quad ".join(f"{lhs} = {to_latex(t, symbols)}" for lhs, t in zip(ACCEL_LATEX[family], trees, strict=True))


def build(family: str, title: str, acceleration: list[str], params: list[FreeParam]) -> FreeModel:
    """Validate and compile a proposal. Raises ExprError with a message the agent can act on."""
    if family not in STATE_VARS:
        raise ExprError(f"free-form models are written for family 'line' or 'plane', not {family!r}")
    state = STATE_VARS[family]
    want = len(state) // 2
    if len(acceleration) != want:
        raise ExprError(f"a {family} model needs {want} acceleration expression(s) ({', '.join(ACCEL_LATEX[family])}), got {len(acceleration)}")
    if len(params) > MAX_PARAMS:
        raise ExprError(f"at most {MAX_PARAMS} parameters")
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
        names.append(p.name)
    trees = [parse(a, set(state) | set(names)) for a in acceleration]
    unused = set(names) - set().union(*(names_used(t) for t in trees))
    if unused:
        raise ExprError(f"parameters {sorted(unused)} are declared but never used")

    fns = [compile_expr(t) for t in trees]
    d = len(state) // 2

    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
        env = dict(p)
        env.update(zip(state, (float(v) for v in s), strict=True))
        return np.array([*s[d:], *(safe_eval(f, env) for f in fns)])

    symbols = {**STATE_LATEX, **{n: param_latex(n) for n in names}}
    latex = r",\quad ".join(f"{lhs} = {to_latex(t, symbols)}" for lhs, t in zip(ACCEL_LATEX[family], trees, strict=True))
    key = json.dumps({"family": family, "acceleration": acceleration,
                      "params": [[p.name, p.lower, p.upper] for p in params]}, sort_keys=True)
    model_id = "custom_" + hashlib.sha256(key.encode()).hexdigest()[:10]
    spec = ModelSpec(model_id, family, title.strip()[:60] or "Free-form law", latex,
                     tuple(Param(p.name, param_latex(p.name), p.unit, p.lower, p.upper,
                                 float(min(max(p.guess, p.lower), p.upper))) for p in params), rhs)
    return FreeModel(spec, family, tuple(acceleration), tuple(params), tuple(to_python(t) for t in trees))


FREE_MODELS: dict[str, FreeModel] = {}


def register(model: FreeModel) -> ModelSpec:
    """Ids are content hashes, so registering the same proposal twice is harmless."""
    FREE_MODELS[model.spec.id] = model
    MODELS[model.spec.id] = model.spec
    return model.spec


def restore(definition: dict[str, Any]) -> ModelSpec:
    return register(build(definition["family"], definition["title"], list(definition["acceleration"]),
                          [FreeParam(**p) for p in definition["params"]]))


def spec_for(fit_state: dict[str, Any]) -> ModelSpec:
    """The model a stored fit refers to, rebuilding a free-form one from its saved definition if needed."""
    model_id = fit_state["model_id"]
    if model_id in MODELS:
        return MODELS[model_id]
    if fit_state.get("free_model"):
        return restore(fit_state["free_model"])
    raise KeyError(f"Unknown model {model_id!r}")
