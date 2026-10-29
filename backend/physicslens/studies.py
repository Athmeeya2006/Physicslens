"""Goal-driven studies: several recordings toward one deliverable, planned and reviewed by the agent, combined by math.

The agent turns a goal ("the hinge damping for my robot sim", "is this fan's bearing wearing out?") into a plan,
asks for recordings, and after each one decides whether the goal is answered. Each recording is verified on its own;
the combination (a weighted mean and an agreement test) and any wear verdict are computed, never claimed.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .agent.reasoning import plan_study, study_turn
from .costguard import CostGuardClient
from .export.mjcf import build_mjcf
from .physics.combine import combine
from .physics.joint import decide_joint, fit_joint
from .physics.models import MODELS
from .physics.scaling import LAWS, Z95, assess_scaling, check_prediction, next_setting

ResultOf = Callable[[str], "dict[str, Any] | None"]
HealthOf = Callable[[list[str]], "dict[str, Any] | None"]
HEALTH_FAMILIES = ("pendulum", "oscillator")


class StudyError(ValueError):
    pass


class Studies:
    def __init__(self, root: Path, client: CostGuardClient):
        self.root = root
        self.client = client
        self._lock = threading.Lock()

    # ---------- storage ----------

    def _path(self, study_id: str) -> Path:
        if not study_id.isalnum():
            raise StudyError("Unknown study.")
        return self.root / study_id / "study.json"

    def get(self, study_id: str) -> dict[str, Any]:
        path = self._path(study_id)
        if not path.is_file():
            raise StudyError("Unknown study.")
        return json.loads(path.read_text())

    def _save(self, study: dict[str, Any]) -> dict[str, Any]:
        path = self._path(study["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(study, indent=1))
        tmp.replace(path)
        return study

    # ---------- the agent's plan and reviews ----------

    def create(self, goal: str, audience: str) -> dict[str, Any]:
        goal = goal.strip()[:400]
        if len(goal) < 8:
            raise StudyError("Describe the goal in a sentence, e.g. 'Is this fan's bearing wearing out?'")
        study_id = uuid.uuid4().hex[:12]
        plan, call = plan_study(self.client, goal, f"study-{study_id}", audience)
        study = {
            "id": study_id, "goal": goal, "created": time.time(), "status": "recording",
            "plan": plan.model_dump(), "recordings": [], "combined": [], "health": None, "turns": [],
            "trail": [{"step": "plan", "provider": call.provider, "model": call.model,
                       "text": f"Plans a {plan.deliverable.replace('_', ' ')}: {'; '.join(plan.quantities)}. "
                               f"{len(plan.recordings)} recording(s). Done when {plan.done_when[0].lower()}{plan.done_when[1:]}",
                       "reasoning_summary": plan.reasoning_summary}],
        }
        return self._save(study)

    def attach(self, study_id: str, run_id: str, label: str, x_value: float | None = None) -> dict[str, Any]:
        with self._lock:
            study = self.get(study_id)
            if study["status"] == "finished":
                raise StudyError("This study is finished; start a new one.")
            if len(study["recordings"]) >= 12:
                raise StudyError("A study holds at most 12 recordings.")
            entry: dict[str, Any] = {"run_id": run_id, "label": label.strip()[:60] or f"Recording {len(study['recordings']) + 1}"}
            if x_value is not None:
                entry["x"] = float(x_value)  # the swept setting for this recording, as the user gave it
            study["recordings"].append(entry)
            return self._save(study)

    def review(self, study_id: str, result_of: ResultOf, health_of: HealthOf, audience: str) -> dict[str, Any]:
        """After a recording finishes: combine everything so far, then let the agent decide what happens next."""
        study = self.get(study_id)
        done = [(r["label"], r["run_id"], result_of(r["run_id"])) for r in study["recordings"]]
        if not done or any(res is None for _, _, res in done):
            return study  # a recording is still being analyzed
        results = [(label, res) for label, _, res in done]
        study["combined"] = combine(results)
        joint = joint_verdict(done)
        if joint is not None and joint != study.get("joint"):
            study["joint"] = joint
            study["trail"].append({"step": "joint_fit", "provider": None, "model": "joint fit across recordings",
                                   "status": joint["status"], "text": joint["text"]})
        accepted = [(label, rid, res) for label, rid, res in done if res.get("status") == "accepted"]
        if study["plan"].get("sweep"):
            before = study.get("sweep") or {}
            study["sweep"] = sweep_state(study, before, {rid: res for _, rid, res in done})
            sw = study["sweep"]
            if sw.get("status") != before.get("status") or len(sw.get("points", [])) != len(before.get("points", [])):
                study["trail"].append({"step": "scaling", "provider": None, "model": "law across recordings",
                                       "status": sw["status"], "text": sweep_text(sw)})
        study["health"] = None
        if (study["plan"]["deliverable"] == "health_check" and len(accepted) >= 2
                and len({res.get("family") for _, _, res in accepted}) == 1 and accepted[0][2].get("family") in HEALTH_FAMILIES):
            study["health"] = health_of([rid for _, rid, _ in accepted])
        context = {
            "goal": study["goal"], "plan": study["plan"],
            "recordings": [{"label": label, "verdict": res.get("status"), "reason": res.get("reason"),
                            "family": res.get("family"), "equation": (res.get("equation") or {}).get("title"),
                            "parameters": [{k: p.get(k) for k in ("name", "value", "std", "unit", "derived")}
                                           for p in res.get("parameters", [])]}
                           for label, _, res in done],
            "combined": [{k: c[k] for k in ("name", "value", "std", "unit", "n", "consistent", "p_value")} for c in study["combined"]],
            "health": study["health"] and {k: study["health"].get(k) for k in ("drift", "summary")},
            "earlier_decisions": [t["message"] for t in study["turns"]],
            "joint_fit": study.get("joint") and {k: study["joint"][k] for k in ("status", "text")},
            "sweep": study.get("sweep") and {k: study["sweep"].get(k) for k in
                                             ("status", "reason", "winner", "expected_law", "next", "checks", "points")},
        }
        turn, call = study_turn(self.client, context, f"study-{study_id}-{len(study['turns'])}", audience)
        entry = {**turn.model_dump(), "model": call.model, "provider": call.provider, "after_recordings": len(done)}
        if turn.action == "finish" and not accepted:
            # The deliverable rests on verified recordings; with none, the study stays open whatever the agent says.
            entry["overruled"] = "No recording has a verified equation yet, so the study stays open."
        study["turns"].append(entry)
        study["status"] = "finished" if turn.action == "finish" and accepted else "recording"
        label = "Finishes the study" if study["status"] == "finished" else "Asks for another recording"
        study["trail"].append({"step": "study_turn", "provider": call.provider, "model": call.model, "action": turn.action,
                               "text": f"{label}: {turn.message}", "reasoning_summary": turn.reasoning_summary})
        return self._save(study)

    # ---------- deliverables ----------

    def mjcf(self, study_id: str, result_of: ResultOf) -> str:
        """The best verified recording's model, with every parameter replaced by the combined value."""
        study = self.get(study_id)
        verified = [res for r in study["recordings"]
                    if (res := result_of(r["run_id"])) and res.get("status") == "accepted" and "fit_state" in res]
        best = min(verified, key=lambda res: res["ranking"][0]["heldout_nrmse"], default=None)
        if best is None:
            raise StudyError("No verified recording to build a model from yet.")
        fs = json.loads(json.dumps(best["fit_state"]))
        for c in study["combined"]:
            if c["name"] in fs["params"] and c["consistent"]:
                fs["params"][c["name"]] = c["value"]
        xml, notes = build_mjcf(fs)
        note = f"PhysicsLens study {study_id}: {study['goal']}. Parameters combined over {len(study['recordings'])} recording(s)."
        notes = [note, *notes]
        return xml.replace("<worldbody>", "<!-- " + " ".join(notes).replace("--", "-") + " -->\n  <worldbody>", 1)

    def report(self, study_id: str, result_of: ResultOf) -> str:
        study = self.get(study_id)
        plan = study["plan"]
        lines = [f"# Study: {study['goal']}", "", f"**Status:** {study['status']}. **Deliverable:** {plan['deliverable'].replace('_', ' ')}.", "",
                 "## Plan", "", f"Measure: {'; '.join(plan['quantities'])}. Done when: {plan['done_when']}", ""]
        lines += [f"- {r['label']}: {r['how_to_film']}" for r in plan["recordings"]]
        lines += ["", "## Recordings", "", "| Recording | Verdict | Equation | Key values |", "|---|---|---|---|"]
        for r in study["recordings"]:
            res = result_of(r["run_id"]) or {}
            vals = ", ".join(f"{p['name']} = {p['value']:.4g} ± {p['std']:.2g} {p['unit']}" for p in res.get("parameters", []))
            lines.append(f"| {r['label']} | {res.get('status', 'running')} | {(res.get('equation') or {}).get('title', '')} | {vals} |")
        if study["combined"]:
            lines += ["", "## Combined", "", "| Quantity | Combined value | Recordings | Agreement |", "|---|---|---|---|"]
            for c in study["combined"]:
                agree = ("single recording" if c["n"] == 1 else
                         f"consistent (p = {c['p_value']:.2f})" if c["consistent"] else f"**disagree** (p = {c['p_value']:.3f})")
                lines.append(f"| {c['name']} | {c['value']:.4g} ± {c['std']:.2g} {c['unit']} | {c['n']} | {agree} |")
        if study["health"]:
            lines += ["", "## Wear check", "", study["health"].get("summary", "")]
        if study["turns"]:
            last = study["turns"][-1]
            lines += ["", "## Conclusion", "", last["message"]]
            if last.get("maintenance_note"):
                lines += ["", f"**Maintenance:** {last['maintenance_note']}"]
            if last.get("next_check"):
                lines += ["", f"**Next check:** {last['next_check']}"]
        models = sorted({t["model"] for t in study["trail"] if t.get("model")})
        lines += ["", "## Provenance", "", f"- Planning and reviews: {', '.join(models)}",
                  "- Each recording: tracked by sam-3.1, equations verified on held-out motion; see its own report.",
                  "- Combined values: inverse-variance weighted means; agreement by a chi-squared test (p < 0.01 = disagree).",
                  "", "Generated by PhysicsLens.", ""]
        return "\n".join(lines)


def joint_verdict(done: list[tuple[str, str, dict[str, Any] | None]]) -> dict[str, Any] | None:
    """When a recording left laws tied, fit every recording of that system together: the closed experiment loop.

    Runs only when at least one recording is ambiguous and two or more recordings of the same family have a fitted
    state. Candidates are the leading laws of each recording, so the experiment the agent asked for gets its say.
    """
    from .export.sim2real import Sim2RealError, measured_coordinates

    usable = [(label, res) for label, _, res in done if res and res.get("fit_state") and res.get("family")]
    if not any(res.get("status") == "ambiguous" for _, res in usable):
        return None
    family = next(res["family"] for _, res in usable if res.get("status") == "ambiguous")
    same = [(label, res) for label, res in usable if res["family"] == family]
    if len(same) < 2:
        return None
    ids: list[str] = []
    for _, res in same:
        top = [r["model_id"] for r in res.get("ranking", [])][:3] if res.get("status") != "accepted" else [res["fit_state"]["model_id"]]
        ids += [m for m in top if m in MODELS and MODELS[m].family == family and m not in ids]
    if len(ids) < 2:
        return None
    try:
        clips = [measured_coordinates(res) for _, res in same]
    except (Sim2RealError, KeyError, ValueError) as e:
        return {"status": "no_fit", "text": f"The recordings could not be fitted together: {e}", "models": []}
    verdict = decide_joint([fit_joint(MODELS[m], clips) for m in ids])
    labels = [label for label, _ in same]

    def errors(f) -> str:
        return ", ".join(f"{e:.0%} on {lab}" for e, lab in zip(f.heldout_nrmse, labels, strict=True))

    models = [{"model_id": f.model_id, "title": MODELS[f.model_id].title, "bic": f.bic, "params": f.params,
               "heldout_nrmse": f.heldout_nrmse, "forecasts_well": f.forecasts_well} for f in verdict["ranked"]]
    if verdict["status"] == "accepted":
        w = verdict["winner"]
        rivals = [f for f in verdict["ranked"] if f is not w]
        why = "; ".join(f"{MODELS[f.model_id].title} {'misses' if not f.forecasts_well else 'fits'} with held-out "
                        f"errors {errors(f)} and is {f.bic - w.bic:.0f} BIC worse" for f in rivals)
        text = (f"Fitted all {len(same)} recordings together with shared physics: {MODELS[w.model_id].title} forecasts "
                f"each one ({errors(w)}). {why}.")
    elif verdict["status"] == "ambiguous":
        text = (f"Even fitted together, {' and '.join(MODELS[f.model_id].title for f in verdict['contenders'])} "
                "still explain the recordings about equally well.")
    else:
        text = "No single law forecasts every recording when they are fitted together."
    return {"status": verdict["status"], "text": text, "models": models, "recordings": labels}


def observable(res: dict[str, Any], response: str) -> tuple[float, float] | None:
    """The swept response from one verified recording: a fitted parameter, or the period derived from one."""
    params = {p["name"].lower(): p for p in res.get("parameters", [])}

    def val(name: str) -> tuple[float, float] | None:
        p = params.get(name.lower())
        if p is None or "assumed" in (p.get("derived") or ""):
            return None  # an assumed value (g as the ruler) is not a measurement
        return float(p["value"]), float(p.get("std") or 0.0)

    if response.strip().lower() != "period":
        return val(response.strip())
    if res.get("family") == "pendulum" and (g := val("g")) and (length := val("L")):
        t = 2 * math.pi * math.sqrt(length[0] / g[0])
        return t, 0.5 * t * math.hypot(g[1] / g[0], length[1] / length[0])
    if res.get("family") == "oscillator" and (w := val("omega")):
        t = 2 * math.pi / w[0]
        return t, t * w[1] / w[0]
    return None


def _setting(rec: dict[str, Any], res: dict[str, Any], variable: str) -> float | None:
    if rec.get("x") is not None:
        return float(rec["x"])
    # A pendulum length typed in for the analysis is the setting itself.
    if "length" in variable.lower():
        p = next((p for p in res.get("parameters", []) if p["name"] == "L" and "given" in (p.get("derived") or "")), None)
        return float(p["value"]) if p else None
    return None


def sweep_state(study: dict[str, Any], before: dict[str, Any], results: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Points from verified recordings, the law code accepts across them, and the next prediction made before filming."""
    plan = study["plan"]["sweep"]
    points = []
    for rec in study["recordings"]:
        res = results.get(rec["run_id"]) or {}
        if not res or res.get("status") != "accepted":
            continue
        x = _setting(rec, res, plan["variable"])
        y = observable(res, plan["response"])
        if x is None or y is None or y[1] <= 0:
            continue
        points.append({"label": rec["label"], "run_id": rec["run_id"], "x": x, "y": y[0], "std": y[1]})
    state: dict[str, Any] = {k: plan[k] for k in ("variable", "unit", "response", "expected_law", "why")}
    state["points"] = points
    # A prediction made earlier is judged by the first recording at that setting that arrived after it.
    checks = list(before.get("checks", []))
    pred = before.get("prediction")
    if pred:
        known = {p["run_id"] for p in before.get("points", [])}
        hit = next((p for p in points if p["run_id"] not in known and math.isclose(p["x"], pred["x"], rel_tol=0.02)), None)
        if hit is not None:
            checks.append({"x": hit["x"], "label": hit["label"], "predicted": pred["y"], "predicted_std": pred["std"],
                           "measured": hit["y"], "measured_std": hit["std"],
                           **check_prediction(pred["y"], pred["std"], hit["y"], hit["std"])})
            pred = None
    state["checks"] = checks
    v = assess_scaling([p["x"] for p in points], [p["y"] for p in points], [p["std"] for p in points])
    state["status"], state["reason"] = v.status, v.reason
    state["winner"] = None if v.winner is None else {
        "law": v.winner.law, "title": LAWS[v.winner.law].title, "latex": LAWS[v.winner.law].latex,
        "params": v.winner.params, "loo_z": v.winner.loo_z, "matches_expectation": v.winner.law == plan["expected_law"]}
    state["ranked"] = [{"law": f.law, "title": LAWS[f.law].title, "bic": f.bic, "worst_loo_z": max(abs(z) for z in f.loo_z)}
                       for f in v.ranked]
    recorded = [p["x"] for p in points]
    planned_left = [x for x in plan["values"] if not any(math.isclose(x, r, rel_tol=0.02) for r in recorded)]
    x_next = planned_left[0] if planned_left else next_setting(v, recorded)
    state["next"] = None
    if v.winner is not None and x_next is not None:
        y, sd = v.winner.predict(x_next)
        state["next"] = {"x": x_next, "y": y, "std": sd, "low": y - Z95 * sd, "high": y + Z95 * sd}
        pred = {"x": x_next, "y": y, "std": sd}
    elif v.status == "ambiguous" and x_next is not None:
        state["next"] = {"x": x_next, "separates": [LAWS[f.law].title for f in v.contenders[:2]]}
    state["prediction"] = pred
    return state


def sweep_text(sw: dict[str, Any]) -> str:
    n = len(sw["points"])
    unit = f" {sw['unit']}" if sw.get("unit") else ""
    if sw["status"] == "too_few":
        text = f"{n} verified recording(s) of {sw['response']} against {sw['variable']}; a law needs at least three settings."
    elif sw["status"] == "accepted":
        w = sw["winner"]
        expect = "as the agent expected" if w["matches_expectation"] else f"not the {sw['expected_law']} the agent expected"
        text = f"Across {n} recordings: {w['title']} ({expect}). {sw['reason']}"
    else:
        text = f"Across {n} recordings: {sw['reason']}"
    nxt = sw.get("next")
    if nxt and "y" in nxt:
        text += (f" Prediction before filming, at {sw['variable']} = {nxt['x']:.3g}{unit}: {sw['response']} = "
                 f"{nxt['y']:.4g} (95%: {nxt['low']:.4g} to {nxt['high']:.4g}).")
    elif nxt:
        text += f" Recording at {sw['variable']} = {nxt['x']:.3g}{unit} would separate {' and '.join(nxt['separates'])}."
    for c in sw.get("checks", [])[-1:]:
        text += (f" The earlier prediction for {c['label']} was {'confirmed' if c['within_95'] else 'missed'}: "
                 f"predicted {c['predicted']:.4g}, measured {c['measured']:.4g} ({c['z']:+.1f} sigma).")
    return text
