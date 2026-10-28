"""Write, check, simulate, repair: Muse Wonder writes a controller, only simulation decides whether it is used.

Each attempt is static-checked (codecheck.py), then run in the sandbox (harness.py) on the nominal identified model
plus bootstrap variants. A failed attempt goes back to the model as a short report; after the last attempt the
result is the honest "no controller met the goal", with the best attempt's numbers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..agent.llm import strict_schema
from ..agent.schemas import ControllerProposal, ControlTask
from ..costguard import CostGuardClient
from ..costguard.errors import CostGuardError
from ..physics.custom import spec_for
from .codecheck import CONTRACT, check_controller
from .harness import SandboxLimits, run_controller, summarize
from .system import (
    GOAL_TEXT,
    build_plants,
    check_supported,
    evaluation_job,
    infeasibility,
    textbook_controller,
)

log = logging.getLogger("physicslens.control")

WONDER_MODEL = "muse-wonder-1.0"
PROMPT_VERSION = "controller-v1"
SIM_LABEL = ("Simulation. Every number here comes from MuJoCo runs of the identified model and its uncertainty "
             "range, not from a real motor.")
FAILED_MESSAGE = "No controller met the goal within the actuator limit."
REPORT_WORST = 3
ERROR_TEXT_MAX = 300

SYSTEM_PROMPT = f"""You write feedback controllers in Python for a physical system that was identified from a phone video.
Your code is tested in MuJoCo simulation on the identified model and on variants drawn from its measured uncertainty.
Only that test decides whether your controller is used.

Contract:
- Define exactly one top-level function: {CONTRACT}
- Return a list with one float: the motor command. For a pendulum it is the torque at the pivot (N m); for an
  oscillator it is the force along the spring axis (N). The moving mass is params["mass"] (1 kg).
- The harness clamps the command to plus or minus params["u_max"]; plan for that limit instead of exceeding it.
- control is called every 2 ms (500 Hz) with the true state. q and qdot are lists with one element.
- Allowed imports: math and numpy only. At module level only simple constants (NAME = literal). Keep any state
  (for example an integrator) in a module-level dict; each variant starts from a fresh copy of your module.
- Not allowed: classes, try, with, global, raise, del, eval, exec, open, getattr and similar, assigning attributes,
  and any name or attribute starting with an underscore. At most 300 lines. Keep it fast: no heavy work per call.

Models (u is your command):
- Pendulum: q[0] is the angle from hanging straight down (rad; pi is upright). A point mass m on a massless rod
  of length L: m L^2 q'' = -m g L sin(q) - m L^2 (gamma q' + beta q'|q'|) + u.
- Oscillator: q[0] is the position along the spring axis (m), with rest position s0:
  m x'' = -m omega^2 (x - s0) - 2 m zeta omega x' + u.

Success, measured by the harness: the error must be within params["tolerance"] by params["time_limit_s"] and
stay inside until time_limit_s + 1 s.
- reach_and_hold: error = q - target (pendulum: angle difference wrapped to [-pi, pi]; oscillator: target position
  is s0 + target).
- swing_up: error = q - pi, wrapped. The motor is usually too weak to lift the pendulum directly.
- stop: error = qdot.
- track_sine: error = q - reference, reference = target sin(2 pi frequency_hz t) (oscillator: s0 + that).
At least 90% of the variants, and the nominal model itself, must succeed. The variants' true parameters differ
from params by the identified uncertainty, so do not rely on exact cancellation.

Reply with JSON: "code" (the complete Python source) and "approach" (one or two plain sentences)."""


def task_key(task: ControlTask, fit_state: dict[str, Any], samples: int, seed: int) -> str:
    """Same task on the same identified model gives the same id, so a solved task is free to open again."""
    blob = json.dumps({"task": task.model_dump(), "model": fit_state["model_id"], "params": fit_state["params"],
                       "ctx": fit_state["ctx"], "y0": fit_state["y0"], "samples": samples, "seed": seed,
                       "prompt": PROMPT_VERSION}, sort_keys=True, default=float)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class LoopConfig:
    max_attempts: int = 3
    success_fraction: float = 0.9
    effort: str = "low"
    max_output_tokens: int = 8000
    limits: SandboxLimits = field(default_factory=SandboxLimits)

    @classmethod
    def from_client(cls, client: CostGuardClient) -> LoopConfig:
        c = client.limits.get("control") or {}
        return cls(int(c.get("max_attempts", 3)), float(c.get("success_fraction", 0.9)), str(c.get("wonder_effort", "low")),
                   int(c.get("wonder_max_output_tokens", 8000)), SandboxLimits.from_config(c))


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4) if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v


def worst_samples(samples: list[dict[str, Any]], params: list[dict[str, float]], k: int = REPORT_WORST) -> list[dict[str, Any]]:
    """Failures first, the largest remaining error first among them."""
    def badness(i: int) -> tuple[int, float]:
        s = samples[i]
        fe = s.get("final_error")
        return (0 if s.get("success") else 1, math.inf if fe is None else float(fe))

    order = sorted(range(len(samples)), key=badness, reverse=True)[:k]
    out = []
    for i in order:
        m = {key: samples[i].get(key) for key in ("success", "settle_s", "overshoot", "final_error", "peak_u", "rms_u",
                                                   "saturation")}
        err = samples[i].get("error")
        out.append({"variant": "nominal" if i == 0 else f"bootstrap {i}", "params": _round(params[i]), **_round(m),
                    **({"error": str(err)[:ERROR_TEXT_MAX]} if err else {})})
    return out


def failure_report(attempt: int, summary: dict[str, Any] | None, worst: list[dict[str, Any]], check_reason: str | None,
                   threshold: float) -> str:
    """What the model sees about its last attempt: compact, numbers only, no paths."""
    if check_reason is not None:
        return f"Attempt {attempt} was rejected before running, by the static check: {check_reason}"
    assert summary is not None
    lines = [(f"Attempt {attempt}: success on {summary['successes']} of {summary['samples']} variants "
              f"(nominal model: {'success' if summary['nominal_success'] else 'failure'}); needed {threshold:.0%} "
              f"including the nominal model."),
             f"Mean command saturation: {_fmt(summary['mean_saturation'])}; mean final error: {_fmt(summary['mean_final_error'])}.",
             "Worst variants (params, then metrics):"]
    lines += [json.dumps(w) for w in worst]
    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def system_text(result: dict[str, Any], task: ControlTask, params: dict[str, float], start: tuple[float, float]) -> str:
    fs = result["fit_state"]
    spec = spec_for(fs)
    eq = result.get("equation") or {}
    stds = {p["name"]: p.get("std") for p in result.get("parameters") or [] if isinstance(p, dict) and "name" in p}
    unc = {k: v for k, v in stds.items() if isinstance(v, int | float)}
    return "\n".join([
        f"System: {spec.title} ({spec.family}). Identified law: {eq.get('latex') or spec.latex}",
        f"Identified parameters (one standard deviation): {json.dumps(_round(unc)) if unc else 'see params'}",
        f"Goal: {task.goal} ({GOAL_TEXT[task.goal]}).",
        f"Start state: q = {start[0]:.4g}, qdot = {start[1]:.4g}.",
        f"params passed to control: {json.dumps(_round(params), sort_keys=True)}",
    ])


def propose(client: CostGuardClient, cfg: LoopConfig, prompt: str, run_id: str, audience: str,
            run_cap_usd: float) -> tuple[ControllerProposal | None, dict[str, Any], str | None]:
    """One paid call. Returns (proposal or None, call info, reason it was unusable)."""
    r = client.meta_text(
        model=WONDER_MODEL, input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        instructions=SYSTEM_PROMPT, schema=strict_schema(ControllerProposal), schema_name="controller",
        effort=cfg.effort, max_output_tokens=cfg.max_output_tokens, prompt_cache_key=f"physicslens-{PROMPT_VERSION}",
        run_id=run_id, audience=audience, run_cap_usd=run_cap_usd, timeout=client.wonder_timeout,
    )
    # Cost stays in the ledger: task results are shown to judges, who never see spend.
    info = {"provider": "meta_api", "model": r.model, "cached": r.cached, "reasoning_summary": r.reasoning}
    text = r.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in text else ""
    try:
        return ControllerProposal.model_validate_json(text), info, None
    except ValidationError as e:
        log.warning("controller answer did not match the schema: %s", e.errors()[:2])
        return None, info, "the answer was not JSON with the fields code and approach"


def _header(result: dict[str, Any], task: ControlTask, params: dict[str, float], summary: dict[str, Any],
            model: str, attempt: int) -> str:
    spec = spec_for(result["fit_state"])
    lines = [
        f"Controller for: {spec.title} ({spec.family}), identified from video by PhysicsLens.",
        f"Identified parameters: {json.dumps(_round(params), sort_keys=True)}",
        (f"Goal: {task.goal}, target {task.target:g}, tolerance {task.tolerance:g}, actuator limit "
         f"{task.actuator_limit:g} per kg, within {task.time_limit_s:g} s (then held 1 s)."),
        (f"Robustness: succeeded on {summary['successes']} of {summary['samples']} simulated variants (nominal model "
         f"plus bootstrap samples of its uncertainty), MuJoCo simulation only."),
        f"Written by {model} (attempt {attempt}); accepted by simulation, not by the model.",
        f"Contract: {CONTRACT}",
    ]
    return "".join(f"# {line}\n" for line in lines) + "\n"


def solve(
    client: CostGuardClient,
    result: dict[str, Any],
    task: ControlTask,
    sample_params: list[dict[str, float]],
    task_dir: Path,
    *,
    ledger_run_id: str,
    audience: str,
    calibrated: bool,
    progress: Callable[[str], None] = lambda s: None,
    cfg: LoopConfig | None = None,
) -> dict[str, Any]:
    """sample_params: nominal first, then the bootstrap samples. Writes every attempt under task_dir."""
    cfg = cfg or LoopConfig.from_client(client)
    check_supported(result, task)
    fs = result["fit_state"]
    task_dir.mkdir(parents=True, exist_ok=True)
    plants = build_plants(fs, task, sample_params)
    job = evaluation_job(fs, task, plants)
    n = len(plants)
    out: dict[str, Any] = {"task": task.model_dump(), "label": SIM_LABEL, "samples": n, "attempts": [], "trail": [],
                           "accepted_attempt": None, "infeasibility_note": infeasibility(fs, task)}

    progress(f"Scoring the textbook controller on {n} variants")
    baseline_code = textbook_controller(fs, task, calibrated)
    baseline = summarize(run_controller(baseline_code, job, cfg.limits), cfg.success_fraction)
    out["baseline"] = {"summary": baseline, "code": baseline_code}

    if out["infeasibility_note"]:
        # A bound on the nominal model already rules the goal out: no paid call can change that.
        out.update(outcome="infeasible", message=f"{FAILED_MESSAGE} {out['infeasibility_note']}")
        _write(task_dir, out)
        return out

    cap = client.guard.budget.get("control_task_cap_usd")
    if cap is None:
        raise CostGuardError("control_task_cap_usd is missing from config/budget.yaml; refusing to call the model.")
    base_prompt = system_text(result, task, job["params"], (plants[0].q0, plants[0].qdot0))
    report: str | None = None
    last_code: str | None = None
    best: dict[str, Any] | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        prompt = base_prompt
        if report is not None:
            shown = f"\n\nYour previous controller:\n```python\n{last_code}\n```" if last_code else ""
            prompt += f"{shown}\n\n{report}\nFix it."
        progress(f"Muse Wonder writes the controller (attempt {attempt} of {cfg.max_attempts})")
        proposal, info, bad = propose(client, cfg, prompt, ledger_run_id, audience, float(cap))
        rec: dict[str, Any] = {"attempt": attempt, "model": info["model"], "cached": info["cached"], "approach": None, "code": None, "check": None,
                               "summary": None, "worst": []}
        out["trail"].append({"step": "controller", "provider": "meta_api", "model": info["model"],
                             "text": proposal.approach if proposal else f"Attempt {attempt}: {bad}.",
                             **({"reasoning_summary": info["reasoning_summary"]} if info["reasoning_summary"] else {})})
        if proposal is None:
            last_code = None
            rec["check"] = {"ok": False, "reason": bad}
            report = f"Attempt {attempt} was unusable: {bad}."
            out["attempts"].append(rec)
            continue
        rec.update(approach=proposal.approach, code=proposal.code)
        (task_dir / f"attempt_{attempt}.py").write_text(proposal.code)
        last_code = proposal.code
        check = check_controller(proposal.code)
        rec["check"] = {"ok": check.ok, "reason": check.reason}
        if not check.ok:
            report = failure_report(attempt, None, [], check.reason, cfg.success_fraction)
            out["trail"].append({"step": "control_test", "provider": None, "model": "MuJoCo simulation",
                                 "label": "Rejects the code before running it", "status": "rejected",
                                 "text": f"Static check: {check.reason}"})
            out["attempts"].append(rec)
            continue
        progress(f"Testing attempt {attempt} on {n} simulated variants")
        samples = run_controller(proposal.code, job, cfg.limits)
        summary = summarize(samples, cfg.success_fraction)
        rec.update(summary=summary, worst=worst_samples(samples, sample_params))
        out["attempts"].append(rec)
        out["trail"].append({"step": "control_test", "provider": None, "model": "MuJoCo simulation",
                             "label": f"Tests it on {n} variants", "status": "accepted" if summary["passed"] else "rejected",
                             "text": f"{summary['successes']} of {n} variants meet the goal "
                                     f"({'nominal model included' if summary['nominal_success'] else 'the nominal model fails'})."})
        if best is None or summary["successes"] > best["summary"]["successes"]:
            best = rec
        if summary["passed"]:
            out.update(outcome="accepted", accepted_attempt=attempt,
                       message=f"Muse Wonder's controller: {summary['successes']} of {n} succeed; textbook controller: "
                               f"{baseline['successes']} of {n}.")
            (task_dir / "controller.py").write_text(
                _header(result, task, job["params"], summary, info["model"], attempt) + proposal.code)
            _write(task_dir, out)
            return out
        report = failure_report(attempt, summary, rec["worst"], None, cfg.success_fraction)

    out.update(outcome="failed", best_attempt=best["attempt"] if best else None, message=FAILED_MESSAGE)
    _write(task_dir, out)
    return out


def _write(task_dir: Path, out: dict[str, Any]) -> None:
    (task_dir / "metrics.json").write_text(json.dumps(out, indent=1, default=str))
