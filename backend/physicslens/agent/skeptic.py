"""The referee: after a law is accepted, the reasoning model proposes alternatives and code tries each one.

The model only names alternatives. Every one is fitted to the same coordinates and judged by the same rule as the
original candidates (`physics/verify.assess`), so the verdict can change only through numerical verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..costguard import CostGuardClient
from ..physics.coords import Coordinates
from ..physics.custom import FreeParam, build, register
from ..physics.expr import ExprError
from ..physics.fit import fit_model
from ..physics.models import MODELS, models_for
from ..physics.verify import Verdict, assess, judge_candidates
from .llm import LLMCall
from .reasoning import skeptic_review
from .schemas import FreeModelIn, Scene, SkepticReview


@dataclass(frozen=True)
class Challenged:
    review: SkepticReview
    call: LLMCall
    verdict: Verdict  # the verdict after every testable challenge was fitted; the old one if none was testable
    info: dict[str, Any]  # what the result JSON and the UI show
    rationales: dict[str, str]


def _brief(f) -> dict[str, Any]:
    return {"model_id": f.model_id, "title": MODELS[f.model_id].title, "params": f.params,
            "heldout_nrmse": f.heldout_nrmse, "bic": f.bic}


def challenge(client: CostGuardClient, coords: Coordinates, verdict: Verdict, family: str, summary: dict[str, Any],
              scene: Scene, run_id: str, audience: str = "dev") -> Challenged:
    assert verdict.status == "accepted" and verdict.winner is not None
    winner = verdict.winner.model_id
    tested = {f.model_id for f in verdict.ranked}
    context = {
        "family": family, "object": scene.moving_object, "accepted_law": _brief(verdict.winner),
        "already_tested": [_brief(f) for f in verdict.ranked],
        "untested_models_in_family": [{"model_id": m.id, "title": m.title} for m in models_for(family)
                                      if m.id not in tested and not m.id.startswith("custom_")],
        "trajectory_summary": summary,
    }
    review, call = skeptic_review(client, context, run_id, audience)

    new_verdict, items, rationales = fit_more(coords, verdict, family, [(c.model_id, c.free_model, c.why) for c in review.challenges],
                                             source="the referee")
    new_ids = [i["model_id"] for i in items if i["status"] != "untestable"]
    survived = new_verdict.status == "accepted" and new_verdict.winner is not None and new_verdict.winner.model_id == winner
    info = {
        "accepted_before": winner,
        "tested": len(new_ids),
        "survived": survived,
        "verdict_after": {"status": new_verdict.status,
                          "winner": new_verdict.winner.model_id if new_verdict.winner else None},
        "challenges": items,
        "concerns": review.concerns,
        "reasoning_summary": review.reasoning_summary,
    }
    return Challenged(review, call, new_verdict, info, rationales)


Proposal = tuple[str | None, FreeModelIn | None, str]  # (listed model id, or a written law), and why


def fit_more(coords: Coordinates, verdict: Verdict, family: str, proposals: list[Proposal],
             source: str) -> tuple[Verdict, list[dict[str, Any]], dict[str, str]]:
    """Fit proposed models next to everything already fitted and judge them all by the same rule.

    Shared by the referee and the agent's own "test these models" tool. Proposals that cannot be fitted to this
    motion are reported as untestable and never touch the verdict.
    """
    tested = {f.model_id for f in verdict.ranked}
    new_ids: list[str] = []
    items: list[dict[str, Any]] = []
    rationales: dict[str, str] = {}
    for model_id, free, why in proposals:
        if model_id is not None:
            spec = MODELS.get(model_id)
            if spec is None or spec.family != family:
                items.append({"model_id": model_id, "title": model_id, "status": "untestable",
                              "why": f"{why} Not a {family} model, so it cannot be fitted to this motion."})
                continue
            if model_id in tested or model_id in new_ids:
                items.append({"model_id": model_id, "title": spec.title, "status": "untestable",
                              "why": f"{why} Already tested on this clip."})
                continue
            new_ids.append(model_id)
            rationales[model_id] = f"{why} (proposed by {source})"
            continue
        assert free is not None
        if family not in ("line", "plane"):
            items.append({"model_id": free.title, "title": free.title, "status": "untestable",
                          "why": f"{why} Written laws apply only to motion along a line or in the plane."})
            continue
        try:
            spec = register(build(family, free.title, free.acceleration, [FreeParam(**p.model_dump()) for p in free.params]))
        except ExprError as e:
            items.append({"model_id": free.title, "title": free.title, "status": "untestable", "why": f"{why} It does not parse: {e}"})
            continue
        new_ids.append(spec.id)
        rationales[spec.id] = f"{why} Written by {source}."
    if not new_ids:
        return verdict, items, rationales
    new_verdict = assess(coords, [*verdict.ranked, *(fit_model(MODELS[m], coords) for m in new_ids)])
    judged = judge_candidates(new_verdict)
    by_id = {f.model_id: f for f in new_verdict.ranked}
    for m in new_ids:
        status, why = judged.get(m, ("rejected", "the fit did not converge"))
        items.append({"model_id": m, "title": MODELS[m].title, "status": status, "why": f"{rationales[m]} Result: {why}",
                      "heldout_nrmse": by_id[m].heldout_nrmse if m in by_id else None,
                      **({"latex": MODELS[m].latex} if m.startswith("custom_") else {})})
    return new_verdict, items, rationales


def outcome_text(info: dict[str, Any], winner_title: str) -> str:
    n = info["tested"]
    if n == 0:
        return f"Tried to break \"{winner_title}\" but proposed nothing code could test."
    if info["survived"]:
        return (f"\"{winner_title}\" survived {n} challenge{'s' if n != 1 else ''}: no alternative did better by the same rule "
                "(forecast the held-out motion, then the best BIC).")
    after = info["verdict_after"]
    if after["status"] == "accepted":
        return (f"A challenger also forecast the held-out motion and fits the whole motion better (BIC): the verdict changes "
                f"to \"{MODELS[after['winner']].title}\".")
    return f"A challenger did as well as \"{winner_title}\" on held-out motion: the verdict is now {after['status']}."
