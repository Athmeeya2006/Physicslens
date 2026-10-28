"""A one-page identification report an engineer can forward: what was measured, how, how sure, and what it rules out."""

from __future__ import annotations

import math
import re
from typing import Any

from ..physics.models import needs_scale

VERDICT = {"accepted": "Verified equation", "ambiguous": "Needs one more experiment", "no_fit": "No clean governing equation"}
SCALE = {"none": "none in view; lengths are in arbitrary units", "user": "the pendulum length given by the user",
         "object_size": "the moving object's size given by the user", "gravity": "Earth's gravity, assumed (no reference in view)"}


def _num(v: float, decimals: int | None = None) -> str:
    a = abs(v)
    if a and (a < 1e-4 or a >= 1e5):
        return f"{v:.3e}"
    return f"{v:.{decimals}f}" if decimals is not None else f"{v:.4g}"


def _decimals(std: float) -> int | None:
    """Two significant figures of the uncertainty decide how many decimals the value deserves."""
    return min(8, max(0, 1 - math.floor(math.log10(std)))) if std > 0 and math.isfinite(std) else None


def _unit(u: str) -> str:
    u = re.sub(r"^1/([a-z]+)$", r"\1^-1", u)
    return u.replace("^-1", "⁻¹").replace("^2", "²").replace("^3", "³")


def report_markdown(result: dict[str, Any], title: str | None = None) -> str:
    r = result
    eq = r.get("equation") or {}
    clip = r.get("clip") or {}
    cal = r.get("calibration") or {}
    lines = [f"# {title or eq.get('title') or VERDICT.get(r.get('status', ''), 'Analysis')}", "",
             f"**Verdict:** {VERDICT.get(r.get('status', ''), r.get('status'))}. {r.get('reason', '')}", ""]

    lines += ["## Setup", "",
              f"- Object: {(r.get('scene') or {}).get('moving_object', 'unknown')}, read as {r.get('family', 'unknown')} motion",
              f"- Clip: {clip.get('width')} by {clip.get('height')} px, {clip.get('frames')} frames at {clip.get('fps', 0):.2f} fps"
              + (f", slow motion x{clip['slow_motion']:g} (all times are real time)" if (clip.get("slow_motion") or 1) > 1 else ""),
              f"- Scale: {SCALE.get(cal.get('source', ''), cal.get('source'))}"
              + (f", {cal['px_per_m']:.1f} px/m, uncertainty {cal['rel_std']:.1%}" if cal.get("source") not in (None, "none") else ""),
              ""]
    tracking = next((t for t in r.get("trail", []) if t.get("step") == "tracking"), None)
    lines += ["## Method", "",
              f"- Tracking: SAM 3.1 segmentation. {tracking['text'] if tracking else ''}".rstrip(),
              ("- Candidate equations proposed by the reasoning model; each was fitted to the first three quarters of the "
               "motion and judged on how well it forecasts the last quarter it never saw, with BIC penalizing extra parameters."),
              "- The language model never accepts an equation: only this numerical verification does.", ""]

    if r.get("parameters"):
        lines += ["## Result", "", f"Equation: `{eq.get('latex', '')}`", "",
                  "| Parameter | Value | 95% interval | Note |", "|---|---|---|---|"]
        for p in r["parameters"]:
            note = p.get("derived") or ("uncalibrated" if p.get("uncalibrated") else
                                        "includes the scale's error" if p.get("includes_scale_error") else "")
            d = _decimals(p["std"])
            lines.append(f"| {p['name']} | {_num(p['value'], d)} ± {_num(p['std'], d)} {_unit(p['unit'])} | "
                         f"{_num(p['ci95'][0], d)} to {_num(p['ci95'][1], d)} | {note} |")
        lines.append("")

    lines += ["## Every candidate tested", "", "| Model | Held-out forecast error | ΔBIC | Outcome |", "|---|---|---|---|"]
    verdicts = {i["model_id"]: (i.get("status"), i.get("why")) for t in r.get("trail", []) if t.get("step") == "verification"
                for i in t.get("items") or []}
    best = min((f["bic"] for f in r.get("ranking", [])), default=0.0)
    for f in r.get("ranking", []):
        status, why = verdicts.get(f["model_id"], ("", ""))
        lines.append(f"| {f['title']} | {f['heldout_nrmse']:.1%} | {f['bic'] - best:+.0f} | {status or ''}{': ' + why if why else ''} |")
    lines.append("")

    decisions = [t for t in r.get("trail", []) if t.get("step") in ("agent", "design", "plausibility", "regime")]
    if decisions:
        lines += ["## Decisions", ""]
        lines += [f"- {t['text']}" for t in decisions]
        lines.append("")

    limits = []
    if cal.get("source") == "none":
        limits.append("No scale: lengths, and g, are in arbitrary units; frequencies, damping and rates are exact.")
    if cal.get("source") == "gravity":
        limits.append("Lengths assume the clip was filmed on Earth (g = 9.81 m/s²).")
    if any(needs_scale(p["unit"]) for p in r.get("parameters", [])) and cal.get("source") == "user":
        limits.append("Lengths rest on the length given by the user.")
    if r.get("mujoco", {}).get("notes"):
        limits += r["mujoco"]["notes"]
    limits.append("Motion is measured in the image plane: the camera should be side-on and still.")
    lines += ["## Limitations", ""] + [f"- {x}" for x in limits] + [""]

    models = sorted({f"{c['model']} ({c['step']})" for c in r.get("llm_calls", [])})
    lines += ["## Provenance", "", "- Tracking: sam-3.1 (Meta Model API)",
              f"- Reasoning: {', '.join(models) if models else 'none'}",
              "- Exports alongside this report: MuJoCo model (.xml), Python simulation (.py), parameters (.json)", "",
              "Generated by PhysicsLens.", ""]
    return "\n".join(lines)
