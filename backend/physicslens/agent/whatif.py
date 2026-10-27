"""Turn a spoken or typed what-if question into parameter changes. The math, not the model, computes the answer."""

from __future__ import annotations

import json
from typing import Any

from ..costguard import CostGuardClient
from ..physics.custom import spec_for
from ..physics.whatif import GEOMETRY_PARAMS, Change, changeable
from .llm import LLMCall, ask_json
from .schemas import WhatIfPlan

WHATIF_SYSTEM = """You translate a user's what-if question about a measured physical system into parameter changes.
You only choose which listed parameter changes and how ("multiply", "set" in the parameter's unit, or "add");
the system is then re-simulated numerically. Use only the listed parameter names. Convert units yourself
(e.g. "30 centimeters" for a length in m is set 0.3). If the question asks for something these parameters
can't express, set understood to false, leave changes empty, and say briefly in reply what you can change."""


def describe_parameters(fit_state: dict[str, Any]) -> list[dict[str, Any]]:
    spec = spec_for(fit_state)
    out = [{"name": p.name, "meaning": p.latex, "unit": p.unit, "current": fit_state["params"][p.name]} for p in spec.params]
    for name, key in GEOMETRY_PARAMS.get(spec.family, {}).items():
        out.append({"name": name, "meaning": "pendulum length", "unit": "m", "current": fit_state["ctx"][key]})
    return out


def interpret(client: CostGuardClient, question: str, fit_state: dict[str, Any], run_id: str,
              audience: str = "dev") -> tuple[WhatIfPlan, list[Change], LLMCall]:
    spec = spec_for(fit_state)
    payload = {"system": spec.title, "equation": spec.latex, "parameters": describe_parameters(fit_state),
               "question": question}
    plan, call = ask_json(client, task="whatif", system=WHATIF_SYSTEM, text=json.dumps(payload), schema_model=WhatIfPlan,
                          run_id=run_id, audience=audience, meta_effort="minimal", max_output_tokens=1200)
    allowed = set(changeable(fit_state))
    changes = [Change(c.parameter, c.operation, c.value) for c in plan.changes if c.parameter in allowed]
    return plan, changes, call


def run_whatif(client: CostGuardClient, question: str, result: dict[str, Any], run_id: str,
               audience: str = "dev") -> dict[str, Any]:
    """Question -> parameter changes (model) -> re-simulation (math) -> ghost points for the clip."""
    from ..physics.whatif import WhatIfError, simulate_whatif

    fit_state = result.get("fit_state")
    if not fit_state:
        return {"understood": False, "reply": "This result has no fitted model to change.", "question": question}
    plan, changes, call = interpret(client, question, fit_state, run_id, audience)
    out: dict[str, Any] = {"question": question, "understood": plan.understood and bool(changes),
                           "description": plan.description, "reply": plan.reply, "model": call.model,
                           "provider": call.provider, "changes": [c.__dict__ for c in changes]}
    if not out["understood"]:
        return out
    seconds = result["clip"]["frames"] / result["clip"]["fps"]
    try:
        out.update(simulate_whatif(fit_state, changes, seconds))
    except WhatIfError as e:
        out.update(understood=False, reply=str(e))
    return out
