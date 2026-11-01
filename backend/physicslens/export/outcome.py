"""The outcome card: what a result means for someone who is not a physicist, and what to do with it.

Built from the result's own numbers by templates, never by the language model: the headline, the action, the files,
what it was checked against, the confidence (from the bootstrap) and a status:
  proven              accepted, and an independent truth lies within the result's 95% interval (or within 1%)
  partial             accepted and verified on motion it never saw, but no independent truth (or it misses it)
  not yet validated   no law was accepted
Also the readable law: the equation with the fitted numbers in it, one plain sentence per term, what the forecast
does without that term (from the ranking, so it was really fitted), and the rejected rivals with their reasons.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Per library law: (the term's LaTeX, what it means in plain words, the law without it). The last one is the nested
# special case the ranking also fitted, so "without this term" is a measured forecast, not a guess.
TERMS: dict[str, list[tuple[str, str, str | None]]] = {
    "pendulum": [(r"-\frac{g}{L}\sin\theta", "gravity pulls it back towards the bottom, harder the further out it swings", None)],
    "pendulum_small_angle": [(r"-\frac{g}{L}\theta", "gravity pulls it back towards the bottom, in proportion to the angle (small swings)", None)],
    "pendulum_linear_damping": [(r"-\frac{g}{L}\sin\theta", "gravity pulls it back towards the bottom", None),
                                (r"-\gamma\dot\theta", "slowed in proportion to its speed (friction at the pivot)", "pendulum")],
    "pendulum_quadratic_damping": [(r"-\frac{g}{L}\sin\theta", "gravity pulls it back towards the bottom", None),
                                   (r"-\beta\dot\theta|\dot\theta|", "slowed by air, more the faster it goes", "pendulum")],
    "harmonic": [(r"-\omega^2 (s - s_0)", "the spring pulls it back to its rest position, harder the further it is stretched", None)],
    "damped_harmonic": [(r"-\omega^2 (s - s_0)", "the spring pulls it back to rest", None),
                        (r"-2\zeta\omega\dot s", "slowed in proportion to its speed (viscous damping)", "harmonic")],
    "oscillator_dry": [(r"-\omega^2 (s - s_0)", "the spring pulls it back to rest", None),
                       (r"-f\,\mathrm{sgn}(\dot s)", "a constant rubbing friction against the motion, whatever the speed", "harmonic")],
    "oscillator_mixed": [(r"-\omega^2 (s - s_0)", "the spring pulls it back to rest", None),
                         (r"-2\zeta\omega\dot s", "slowed in proportion to its speed", "oscillator_dry"),
                         (r"-f\,\mathrm{sgn}(\dot s)", "a constant rubbing friction against the motion", "damped_harmonic")],
    "duffing": [(r"-\omega^2 (s - s_0)", "the spring pulls it back to rest", None),
                (r"-\beta (s - s_0)^3", "the spring stiffens (or softens) when stretched far", "damped_harmonic"),
                (r"-2\zeta\omega\dot s", "slowed in proportion to its speed", None)],
    "free_fall": [(r"-g", "gravity pulls it down at a constant rate; nothing else acts", None)],
    "linear_drag": [(r"-g\hat{y}", "gravity pulls it down", None),
                    (r"-k\mathbf{v}", "air slows it in proportion to its speed", "free_fall")],
    "quadratic_drag": [(r"-g\hat{y}", "gravity pulls it down", None),
                       (r"-c|\mathbf{v}|\mathbf{v}", "air slows it, more the faster it goes (with the square of speed)", "free_fall")],
    "spin_dry": [(r"-c\,\mathrm{sgn}(\dot\varphi)", "a constant rubbing friction slows the spin", "spin_free")],
    "spin_viscous": [(r"-k\dot\varphi", "slowed in proportion to how fast it spins (oil, a viscous bearing)", "spin_free")],
    "spin_air": [(r"-q\dot\varphi|\dot\varphi|", "air slows the spin, more the faster it turns", "spin_free")],
    "spin_dry_viscous": [(r"-c\,\mathrm{sgn}(\dot\varphi)", "a constant rubbing friction", "spin_viscous"),
                         (r"-k\dot\varphi", "plus a slowing in proportion to the spin", "spin_dry")],
    "spin_spring": [(r"-\kappa (\varphi - \varphi_0)", "a torsion spring turns it back towards its rest angle", None),
                    (r"-k\dot\varphi", "slowed in proportion to how fast it turns", None)],
    "line_constant_velocity": [(r"0", "nothing pushes or slows it: it keeps a constant speed", None)],
    "line_constant_accel": [(r"a_0", "a constant push (or brake): speed changes at a steady rate", "line_constant_velocity")],
    "plane_constant_velocity": [(r"0", "nothing pushes or slows it: a straight line at constant speed", None)],
    "plane_constant_accel": [(r"a_x, a_y", "a constant push in one direction", "plane_constant_velocity")],
}

FAMILY_NOUN = {"pendulum": "swing", "oscillator": "oscillation", "projectile": "flight", "rotation": "spin",
               "line": "motion", "plane": "motion", "bounce": "bounces", "double_pendulum": "motion"}
EXPORTS = {"pendulum": ["json", "py", "mjcf", "urdf", "sdf", "report"], "oscillator": ["json", "py", "mjcf", "urdf", "sdf", "report"],
           "projectile": ["json", "py", "mjcf", "urdf", "sdf", "report"], "rotation": ["json", "py", "mjcf", "report"],
           "plane": ["json", "py", "report"], "line": ["json", "py", "report"]}


def _fmt(v: float, s: float | None = None) -> str:
    """As many significant figures as the uncertainty supports (two digits of it), else four."""
    if s and s > 0 and math.isfinite(s) and v != 0 and math.isfinite(v):
        sig = max(1, min(10, math.floor(math.log10(abs(v))) - math.floor(math.log10(s)) + 2))
        return f"{v:.{sig}g}"
    return f"{v:.4g}"


def _why_not(r: dict[str, Any], winner_bic: float | None) -> str:
    """Why a rival lost, from its own measured numbers (the verdict's rule, stated plainly)."""
    if r.get("why"):
        return r["why"]
    held = r.get("heldout_nrmse")
    if isinstance(held, (int, float)) and math.isfinite(held) and held > 0.15:
        return f"its forecast of the held-out quarter missed by {held:.0%}"
    if winner_bic is not None and isinstance(r.get("bic"), (int, float)) and math.isfinite(r["bic"]):
        d = r["bic"] - winner_bic
        if d >= 10:
            return f"forecast too, but fits the whole motion {d:.0f} BIC worse"
        return f"within {d:.0f} BIC of the winner: the simpler or exact law was kept"
    return "did not fit"


def numeric_latex(latex: str, params: list[dict[str, Any]]) -> str:
    """The law with its fitted values written in (each parameter symbol replaced by its number)."""
    out = latex
    for p in sorted(params, key=lambda q: -len(q.get("latex") or q["name"])):
        sym = p.get("latex") or p["name"]
        val = _fmt(p["value"], p.get("std"))
        if sym.startswith("\\"):
            out = re.sub(re.escape(sym) + r"(?![A-Za-z])", lambda _m, v=val: "{" + v + "}", out)
        else:
            # A one-letter symbol only where it stands alone (not inside \frac, \sin or another word).
            out = re.sub(r"(?<![\\A-Za-z])" + re.escape(sym) + r"(?![A-Za-z])", lambda _m, v=val: "{" + v + "}", out)
    return out


def readable(result: dict[str, Any]) -> dict[str, Any] | None:
    eq = result.get("equation")
    if not eq:
        return None
    model = eq["model_id"]
    held = {r["model_id"]: r.get("heldout_nrmse") for r in result.get("ranking", [])}
    terms = [{"latex": tex, "text": text, "without": without, "without_title": next(
        (r["title"] for r in result.get("ranking", []) if r["model_id"] == without), None),
        "without_heldout": held.get(without) if without else None}
        for tex, text, without in TERMS.get(model, [])]
    win_bic = next((r.get("bic") for r in result.get("ranking", []) if r["model_id"] == model), None)
    rivals = [{"title": r["title"], "why": _why_not(r, win_bic), "heldout": r.get("heldout_nrmse")}
              for r in result.get("ranking", []) if r["model_id"] != model][:5]
    return {"latex_numeric": numeric_latex(eq["latex"], result.get("parameters", [])),
            "heldout": held.get(model), "terms": terms,
            "written_by_agent": model.startswith("custom_"), "rivals": rivals}


def _confidence(result: dict[str, Any]) -> str | None:
    params = [p for p in result.get("parameters", []) if p.get("std")]
    if not params:
        return None
    reps = result.get("bootstrap_reps")
    lead = params[0]
    lo, hi = lead.get("ci95") or (lead["value"] - 1.96 * lead["std"], lead["value"] + 1.96 * lead["std"])
    unit = f" {lead.get('unit')}" if lead.get("unit") else ""
    return (f"{lead['name']} between {_fmt(lo, lead['std'])} and {_fmt(hi, lead['std'])}{unit} (95%"
            + (f", from {reps} bootstrap refits" if reps else "") + ")")


def _agreed_confidence(result: dict[str, Any]) -> str | None:
    agreed = result.get("agreed_parameters") or []
    if not agreed:
        return None
    a = agreed[0]
    lo, hi = a["ci95"]
    unit = f" {a['unit']}" if a.get("unit") and not a.get("uncalibrated") else ""
    return (f"{a['name']} between {_fmt(lo, a['std'])} and {_fmt(hi, a['std'])}{unit} (95%, spanning every tied law's "
            "bootstrap interval)")


def build_outcome(result: dict[str, Any], truth: dict[str, Any] | None = None, action: str | None = None,
                  artifacts: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """truth: {"name", "value", "source"} for the parameter an independent measurement exists for."""
    status = result.get("status")
    eq = result.get("equation") or {}
    family = result.get("family", "")
    noun = FAMILY_NOUN.get(family, "motion")
    held = next((r.get("heldout_nrmse") for r in result.get("ranking", []) if r["model_id"] == eq.get("model_id")), None)
    checked = None
    outcome_status = "not yet validated"
    if status == "accepted":
        headline = (f"The {noun} follows one law, {eq.get('title', 'the fitted law').lower()}: fitted on the first three "
                    f"quarters, it predicted the last quarter"
                    + (f" within {held:.1%}" if isinstance(held, (int, float)) and math.isfinite(held) else "") + ".")
        outcome_status = "partial"
        if truth:
            p = next((q for q in result.get("parameters", []) if q["name"] == truth["name"]), None)
            if p is not None:
                err = (p["value"] - truth["value"]) / truth["value"]
                lo, hi = p.get("ci95") or (p["value"] - 1.96 * p["std"], p["value"] + 1.96 * p["std"])
                inside = lo <= truth["value"] <= hi or abs(err) <= 0.01
                checked = {"name": truth["name"], "truth": truth["value"], "measured": p["value"],
                           "relative_error": err, "within_interval": bool(lo <= truth["value"] <= hi),
                           "source": truth.get("source", "")}
                outcome_status = "proven" if inside else "partial"
        default_action = ("Download the model and use it in your simulator or code; its numbers come with intervals."
                          if family in EXPORTS else "Use the fitted law and its intervals; the report has every step.")
    elif status == "ambiguous":
        headline = f"Two laws describe this {noun} equally well; the data cannot tell them apart yet."
        agreed = result.get("agreed_parameters") or []
        if agreed:
            a = agreed[0]
            unit = f" {a['unit']}" if a.get("unit") and not a.get("uncalibrated") else ""
            headline += f" Both give {a['name']} = {_fmt(a['value'], a['std'])} ± {_fmt(a['std'], a['std'])}{unit}, so that is measured either way."
            if truth:
                p = next((q for q in agreed if q["name"] == truth["name"]), None)
                if p is not None:
                    lo, hi = p["ci95"]
                    checked = {"name": truth["name"], "truth": truth["value"], "measured": p["value"],
                               "relative_error": (p["value"] - truth["value"]) / truth["value"],
                               "within_interval": bool(lo <= truth["value"] <= hi), "source": truth.get("source", "")}
        default_action = ((result.get("agent") or {}).get("message")
                          or "Record again with a larger motion, so the laws make different predictions.")
    else:
        headline = f"No single law predicts this {noun}: PhysicsLens says so rather than guess."
        default_action = "Record again: longer, steadier, side-on, with the whole motion in view."
    files = artifacts if artifacts is not None else [{"kind": k} for k in EXPORTS.get(family, ["report"])] if status == "accepted" else []
    return {"headline": headline, "action": action or default_action, "artifacts": files, "checked_against": checked,
            "confidence": _confidence(result) if status == "accepted" else _agreed_confidence(result), "status": outcome_status,
            "readable": readable(result) if status == "accepted" else None}
