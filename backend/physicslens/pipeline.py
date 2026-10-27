"""Recorded mode end to end: video -> scene -> SAM tracks -> calibrated trajectory -> hypotheses -> verified equation."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import meta_sam_parser as msp
import numpy as np

from .agent.reasoning import agent_turn, propose_hypotheses, understand_scene
from .agent.schemas import Scene
from .costguard import CostGuardClient
from .costguard.cache import file_digest
from .export.mjcf import MjcfError, check_against_fit
from .perception import media
from .perception.calibration import (
    Calibration,
    CalibrationError,
    from_known_length,
    from_reference,
)
from .perception.sam_track import ObjectTrack, TrackingError, tracks_from_text
from .physics.collisions import BounceError, detect_bounces, fit_bounces
from .physics.coords import Coordinates, fit_circle, to_coordinates
from .physics.design import design_experiment
from .physics.double import HORIZON_S as DOUBLE_HORIZON_S
from .physics.double import DoublePendulumError, fit_double, to_double_coords
from .physics.double import simulate as double_simulate
from .physics.double import summary as double_summary
from .physics.fit import FitResult, fit_model
from .physics.models import MODELS, models_for
from .physics.regime import RegimeChange, detect_change
from .physics.rotation import RotationError, rotation_coordinates
from .physics.sindy import discover
from .physics.uncertainty import bootstrap
from .physics.verify import Verdict, assess, judge_candidates
from .sigproc.cleanup import detect_impacts, differentiate, fill_short_gaps, split_at
from .sigproc.summary import summarize
from .sigproc.trajectory import Trajectory, from_pixels

log = logging.getLogger("physicslens.pipeline")

NOMINAL_PX_PER_M = 1000.0  # used only when nothing in the clip gives a scale
SCALE_NOTE = ("No reference of known size was found, so lengths and g are in arbitrary units. Frequency, damping "
              "and friction rates don't depend on scale and are exact. For a calibrated result, put an A4 sheet "
              "flat in the plane of motion or enter the pendulum length.")
_LENGTH_UNIT = re.compile(r"(^|[^a-z])m($|[^a-z])")


def needs_scale(unit: str) -> bool:
    """True for anything measured in meters (m, m/s^2, 1/m); angles, times and rates are scale-free."""
    return bool(_LENGTH_UNIT.search(unit or ""))

BOUNCE_LATEX = r"\ddot y = -g \ \text{between impacts},\qquad v_y^{+} = -e\, v_y^{-}"

MOVING_ID, REFERENCE_ID, SECOND_ID = "1", "2", "3"
SAM_METADATA = {"mask_encoding": "one_bit", "include_confidence": "true", "propagation_direction": "forward"}
MAX_CLIP_SECONDS = {"dev": 20.0, "judge": 30.0, "public": 10.0}


class AnalysisError(RuntimeError):
    """A problem with the input the user can fix (message is shown as-is)."""


@dataclass(frozen=True)
class RunOptions:
    audience: str = "dev"
    pendulum_length_m: float | None = None  # alternative scale source when no reference object is visible


Progress = Callable[[str], None]


def analyze(client: CostGuardClient, video: Path, run_dir: Path, run_id: str, options: RunOptions | None = None,
            progress: Progress = lambda _msg: None) -> dict[str, Any]:
    options = options or RunOptions()
    run_dir.mkdir(parents=True, exist_ok=True)
    limits = client.limits
    media_cfg = limits.get("media") or {}
    max_turns = int((limits.get("agent") or {}).get("max_turns", 4))
    llm_calls: list[dict[str, Any]] = []
    # What the agent and the verifier decided, in order, each tagged with who decided it (shown on screen).
    trail: list[dict[str, Any]] = []

    def record(step: str, call, text: str, **extra: Any) -> None:
        llm_calls.append({"step": step, "provider": call.provider, "model": call.model, "cached": call.cached,
                          "reasoning": call.reasoning})
        trail.append({"step": step, "provider": call.provider, "model": call.model, "text": text, **extra})

    progress("Preparing video")
    clip = run_dir / "clip.mp4"
    info = media.prepare_sam_clip(video, clip, max_fps=float(media_cfg.get("sam_max_fps_recorded", 30)),
                                  max_seconds=MAX_CLIP_SECONDS[options.audience],
                                  max_long_side=int(media_cfg.get("dev_max_resolution", 1080)))
    min_frames = int(media_cfg.get("sam_min_frames_per_request", 50))
    if info.frames < min_frames:
        raise AnalysisError(f"The clip is too short: at least {min_frames} frames (about 2 seconds) are needed.")

    progress("Understanding the scene")
    frames_jpeg = media.keyframes(clip, int(media_cfg.get("llm_keyframes_max", 4)), int(media_cfg.get("llm_image_long_side_px", 768)))
    scene, call = understand_scene(client, frames_jpeg, run_id, options.audience)
    ref = "no reference object" if scene.reference_kind == "none" else f"reference: {scene.reference_kind.replace('_', ' ')}"
    record("scene", call, f"Sees a {scene.moving_object} moving like a {scene.motion_family}; {ref}.")

    progress(f"Tracking the {scene.moving_object} with SAM 3.1")
    tracks = track_objects(client, clip, info, scene, run_id, options.audience)
    moving = tracks.get(MOVING_ID)
    if moving is None or len(moving.frames) < min_frames // 2:
        raise AnalysisError(f"SAM could not follow the {scene.moving_object} through the clip.")
    trail.append({"step": "tracking", "provider": "meta_api", "model": "sam-3.1",
                  "text": f"Tracked the {scene.moving_object} in {len(moving.frames)} of {info.frames} frames."})

    progress("Calibrating scale")
    calibration = calibrate(scene, tracks.get(REFERENCE_ID), moving, options)
    full, full_origin = full_track(moving, info.fps, calibration.px_per_m)
    traj, origin_px = longest_flight(full, full_origin, calibration.px_per_m)
    rotation: Coordinates | None = None
    if scene.motion_family == "rotation":
        progress("Measuring the spin from the mask orientation")
        rotation, traj, origin_px = rotation_track(moving, info.fps, calibration.px_per_m)

    summary = rotation_summary(rotation) if rotation is not None else summarize(traj)
    progress("Proposing hypotheses")
    hyp, call = propose_hypotheses(client, summary, scene, run_id, options.audience)
    record("hypotheses", call, f"Proposes {len(hyp.candidates)} {hyp.family} models to test.", items=_hypothesis_items(hyp))

    if hyp.family == "double_pendulum":
        second = tracks.get(SECOND_ID)
        if second is None:
            raise AnalysisError("A double pendulum needs both bobs tracked, and SAM did not find the outer one.")
        progress("Fitting the double pendulum with short-horizon predictions")
        dbl, ghosts, steps = double_analysis(moving, second, info.fps, calibration.px_per_m)
        trail.extend(steps)
        status = "accepted" if dbl["accepted"] else ("ambiguous" if not dbl["length_reliable"] else "no_fit")
        progress("Agent reviewing the verdict")
        context = {"verdict": status, "verdict_reason": dbl["reason"], "family": "double_pendulum",
                   "object": scene.moving_object, "trajectory_summary": summary,
                   "double_pendulum": {k: dbl[k] for k in ("params", "L1", "L2", "chaotic", "lyapunov_per_s", "predictability_s")}}
        turn, call = agent_turn(client, context, run_id, options.audience)
        record("agent", call, turn.message, action=turn.action, reasoning_summary=turn.reasoning_summary)
        progress("Done")
        return {
            "status": status, "reason": dbl["reason"], "family": "double_pendulum", "families_tried": ["double_pendulum"],
            "scene": scene.model_dump(),
            "clip": {"width": info.width, "height": info.height, "fps": info.fps, "frames": info.frames},
            "calibration": {"px_per_m": calibration.px_per_m, "rel_std": calibration.rel_std, "source": calibration.source},
            "agent": {"action": turn.action, "message": turn.message, "reasoning_summary": turn.reasoning_summary},
            "ranking": [], "real_track_px": dbl.pop("inner_track_px"), "object_size_px": float(np.median(moving.long_px)),
            "llm_calls": llm_calls, "trail": trail, "summary": summary, "ghosts": ghosts, "double": dbl,
            **({"scale_note": SCALE_NOTE} if calibration.source == "none" else {}),
        }

    tried: list[str] = []
    family, candidate_ids = hyp.family, _with_control(hyp.family, [h.model_id for h in hyp.candidates])
    rationales = {h.model_id: h.rationale for h in hyp.candidates}
    verdict: Verdict | None = None
    coords: Coordinates | None = None
    turn = None
    change: RegimeChange | None = None
    regimes: list[dict[str, Any]] = []
    bounces: dict[str, Any] | None = None
    for _ in range(max_turns):
        tried.append(family)
        progress(f"Fitting and simulating {len(candidate_ids)} {family} models")
        coords, verdict = fit_and_verify(traj, family, candidate_ids, rotation if family == "rotation" else None)
        judged = judge_candidates(verdict)
        trail.append({"step": "verification", "provider": None, "model": "simulate-and-compare", "text": verdict.reason,
                      "status": verdict.status,
                      "items": [{"model_id": m, "title": MODELS[m].title, "status": st, "why": why} for m, (st, why) in judged.items()]})
        if family == "projectile" and bounces is None:
            progress("Analyzing bounces")
            bounces = analyze_bounces(full, full_origin, calibration.px_per_m)
            if bounces is not None:
                trail.append(bounce_trail_step(bounces))
        design = None
        if verdict.status == "ambiguous" and coords is not None and len(verdict.contenders) >= 2:
            progress("Designing the experiment that would separate the tied models")
            design = design_experiment(coords, list(verdict.contenders))
            if design is not None:
                trail.append(design_trail_step(design))
        progress("Checking whether the dynamics change during the clip")
        change, regimes = (None, []) if family == "rotation" else find_regimes(traj, family, candidate_ids, origin_px, calibration.px_per_m)
        if change is not None:
            trail.append(regime_trail_step(change, regimes))
        progress("Agent reviewing the verdict")
        context = agent_context(verdict, family, summary, scene, tried)
        if calibration.source == "none":
            context["scale"] = "none: lengths and g are uncalibrated, frequency, damping and friction are exact"
        if design is not None:
            context["experiment_design"] = {k: design[k] for k in ("recommended", "decisive", "noise_mm", "models")}
        if bounces is not None:
            context["bounces"] = {k: bounces[k] for k in ("g", "g_std", "e_mean", "e_std", "count", "accepted", "forecast_nrmse")}
        if change is not None:
            context["regime_change"] = {"at_s": round(change.t_estimate, 2), "regimes": [
                {k: r[k] for k in ("start_s", "end_s", "status", "title", "params")} for r in regimes]}
        turn, call = agent_turn(client, context, run_id, options.audience)
        record("agent", call, turn.message, action=turn.action, reasoning_summary=turn.reasoning_summary)
        if verdict.status != "no_fit" or turn.action != "try_family" or not turn.next_family or turn.next_family in tried:
            break
        hyp, call = propose_hypotheses(client, summary, scene, run_id, options.audience, exclude_families=tried)
        record("hypotheses", call, f"Switches to {hyp.family} models.", items=_hypothesis_items(hyp))
        if hyp.family in tried:
            break
        family, candidate_ids = hyp.family, _with_control(hyp.family, [h.model_id for h in hyp.candidates])
        rationales.update({h.model_id: h.rationale for h in hyp.candidates})

    assert verdict is not None and turn is not None
    result: dict[str, Any] = {
        "status": verdict.status,
        "reason": verdict.reason,
        "family": family,
        "families_tried": tried,
        "scene": scene.model_dump(),
        "clip": {"width": info.width, "height": info.height, "fps": info.fps, "frames": info.frames},
        "calibration": {"px_per_m": calibration.px_per_m, "rel_std": calibration.rel_std, "source": calibration.source},
        "agent": {"action": turn.action, "message": turn.message, "reasoning_summary": turn.reasoning_summary},
        "ranking": [fit_summary(f, rationales.get(f.model_id)) for f in verdict.ranked],
        "real_track_px": _track_points(traj.frames, traj.xy, origin_px, calibration.px_per_m),
        # For a spin the ghost is a small marker on the rim, not the whole object.
        "object_size_px": float(np.median(moving.long_px)) * (0.25 if rotation is not None else 1.0),
        "llm_calls": llm_calls,
        "trail": trail,
        "summary": summary,
    }
    if coords is not None:
        try:
            result["sindy"] = discover(coords).equations
        except (ValueError, np.linalg.LinAlgError) as e:  # supporting evidence only; must not sink the result
            log.warning("SINDy failed: %s", e)
            result["sindy"] = None
        result["geometry"] = coords.ctx
        result["ghosts"] = [ghost(coords, f, traj, origin_px, calibration.px_per_m) for f in verdict.contenders]
    if design is not None:
        result["experiment_design"] = design
    if bounces is not None:
        result["bounces"] = {k: v for k, v in bounces.items() if k != "ghost"}
        result.setdefault("ghosts", []).insert(0, {"model_id": "bounce model", "points": bounces["ghost"]})
    if change is not None:
        result["regime_change"] = {"t_estimate": change.t_estimate, "t_safe": change.t, "error_ratio": change.error_ratio}
        result["regimes"] = [{k: v for k, v in r.items() if k != "ghost"} for r in regimes]
        points = [p for r in regimes for p in r["ghost"]]
        if points:
            # One ghost made of each regime's own fit: it should hug the object on both sides of the change.
            result.setdefault("ghosts", []).insert(0, {"model_id": "fit per regime", "points": points})
    lead = verdict.winner or (verdict.contenders[0] if verdict.contenders else None)
    if lead is not None and coords is not None:
        # Everything a what-if needs to re-simulate from the measured starting state and draw it on the clip.
        result["fit_state"] = {
            "model_id": lead.model_id, "params": lead.params, "y0": [float(v) for v in lead.y0],
            "ctx": {k: float(v) for k, v in coords.ctx.items()}, "fps": traj.fps,
            "origin_px": [float(origin_px[0]), float(origin_px[1])], "px_per_m": calibration.px_per_m,
            "first_frame": int(traj.frames[0]),
        }
    if "fit_state" in result:
        # Run the MuJoCo export from the measured state: proof the exported model is the fitted one.
        try:
            result["mujoco"] = check_against_fit(result["fit_state"], len(traj) / traj.fps)
        except MjcfError as e:
            log.warning("MuJoCo check skipped: %s", e)
    if verdict.status == "accepted" and verdict.winner and coords is not None:
        progress("Estimating uncertainty")
        spec = MODELS[verdict.winner.model_id]
        est = bootstrap(spec, coords, verdict.winner)
        result["equation"] = {"model_id": spec.id, "title": spec.title, "latex": spec.latex}
        result["parameters"] = [
            {"name": p.name, "latex": p.latex, "unit": p.unit, "value": est[p.name].value, "std": est[p.name].std,
             "ci95": [est[p.name].ci_low, est[p.name].ci_high]}
            for p in spec.params
        ] + derived_parameters(coords)
    if calibration.source == "none":
        result["scale_note"] = SCALE_NOTE
        for p in result.get("parameters", []):
            p["uncalibrated"] = needs_scale(p["unit"])
    progress("Done")
    return result


def _common_frames(a: ObjectTrack, b: ObjectTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longest run of consecutive frames where both objects are tracked (short gaps filled)."""
    fa, ca = max(fill_short_gaps(a.frames, a.centers_px, max_gap=3), key=lambda s: len(s[0]))
    fb, cb = max(fill_short_gaps(b.frames, b.centers_px, max_gap=3), key=lambda s: len(s[0]))
    common = np.intersect1d(fa, fb)
    if len(common) < 2:
        raise AnalysisError("The two bobs were never tracked at the same time.")
    breaks = np.nonzero(np.diff(common) > 1)[0] + 1
    run = max(np.split(common, breaks), key=len)
    return run, ca[np.searchsorted(fa, run)], cb[np.searchsorted(fb, run)]


def double_analysis(inner: ObjectTrack, outer: ObjectTrack, fps: float, ppm: float) -> tuple[dict[str, Any], list[dict], list[dict]]:
    frames, ci, co = _common_frames(inner, outer)
    origin = ci[0]

    def meters(c: np.ndarray) -> np.ndarray:  # both bobs in ONE frame: relative geometry is the whole point
        return np.column_stack([(c[:, 0] - origin[0]) / ppm, -(c[:, 1] - origin[1]) / ppm])

    t = (frames - frames[0]) / fps
    tr_i, tr_o = Trajectory(t, meters(ci), frames, fps), Trajectory(t, meters(co), frames, fps)
    try:
        c = to_double_coords(tr_i, tr_o)
        fit = fit_double(c)
    except DoublePendulumError as e:
        raise AnalysisError(str(e)) from e
    out = double_summary(fit, c)

    def px(xy: np.ndarray) -> list[list[float]]:
        return [[int(f), round(float(origin[0] + x * ppm), 2), round(float(origin[1] - y * ppm), 2)]
                for f, (x, y) in zip(frames, xy, strict=True)]

    kin = differentiate(c.q, fps)
    states = np.column_stack([kin.pos, kin.vel])
    anchored = reanchored_prediction(fit.params, c, states, every_s=DOUBLE_HORIZON_S)
    free = double_simulate(fit.params, c.L1, c.L2, states[0], t)
    a_inner, a_outer = c.to_xy(anchored)
    every = f"restarted every {DOUBLE_HORIZON_S:.1f} s"
    ghosts = [{"model_id": f"inner bob (model, {every})", "points": px(a_inner)},
              {"model_id": f"outer bob (model, {every})", "points": px(a_outer)}]
    if free is not None:
        ghosts.append({"model_id": "outer bob (one prediction from t = 0)", "points": px(c.to_xy(free[:, :2])[1])})
    out["inner_track_px"] = px(tr_i.xy)
    out["outer_track_px"] = px(tr_o.xy)

    status = "accepted" if fit.accepted else "rejected"
    lam = fit.lyapunov_per_s
    chaos_text = (f"Chaotic: two almost identical starts separate e-fold every {1 / lam:.2f} s (lambda = {lam:.1f}/s). "
                  f"With this tracking noise the motion can be forecast for about {fit.predictability_s:.1f} s; beyond that "
                  f"no model can, so the single long prediction drifts away while the restarted one stays on."
                  if fit.chaotic else
                  f"Not chaotic at this energy (lambda = {lam:.2f}/s): small differences in the start stay small, "
                  f"so the motion stays predictable.")
    steps = [
        {"step": "verification", "provider": None, "model": "simulate-and-compare", "status": "accepted" if fit.accepted else "no_fit",
         "text": fit.reason, "items": [{"model_id": "double_pendulum", "title": "Double pendulum (point masses)",
                                        "status": status, "why": (f"g = {fit.params['g']:.3f} m/s^2, mass ratio {fit.params['mu']:.2f}, "
                                                                  f"L1 = {c.L1:.3f} m, L2 = {c.L2:.3f} m")}]},
        {"step": "chaos", "provider": None, "model": "simulate-and-compare", "text": chaos_text},
    ]
    return out, ghosts, steps


def reanchored_prediction(params: dict[str, float], c, states: np.ndarray, every_s: float) -> np.ndarray:
    """Model prediction restarted from the measured state every few tenths of a second (shows short-term skill)."""
    n, step = len(c.t), max(1, int(every_s * c.fps))
    out = np.array(c.q, dtype=float)
    for i in range(0, n, step):
        j = min(n, i + step)
        if j - i < 2:
            break  # a single trailing sample keeps its measured value
        sim = double_simulate(params, c.L1, c.L2, states[i], c.t[i:j])
        if sim is not None:
            out[i:j] = sim[:, :2]
    return out


def rotation_track(moving: ObjectTrack, fps: float, ppm: float) -> tuple[Coordinates, Trajectory, np.ndarray]:
    """Spin angle from mask orientation, plus the rim marker's path as the 'track' (absolute meters, image origin)."""
    try:
        coords, frames = rotation_coordinates(moving.frames, moving.angles, moving.centers_px, moving.long_px,
                                              moving.short_px, fps, ppm)
    except RotationError as e:
        raise AnalysisError(str(e)) from e
    return coords, Trajectory(coords.t, coords.to_xy(coords.q), frames, fps), np.array([0.0, 0.0])


def rotation_summary(c: Coordinates) -> dict[str, Any]:
    """What the reasoning model sees for a spin: angular speed over time, not positions."""
    kin = differentiate(c.q, 1.0 / (c.t[1] - c.t[0]))
    w = kin.vel[:, 0]
    q = [int(len(w) * f) for f in (0.0, 0.25, 0.5, 0.75)] + [len(w) - 1]
    return {
        "kind": "rotation", "duration_s": round(float(c.t[-1]), 3), "turns": round(float(abs(c.q[-1, 0] - c.q[0, 0]) / (2 * np.pi)), 2),
        "angular_speed_rad_s_at_0_25_50_75_100_percent": [round(float(w[i]), 3) for i in q],
        "elongation": round(c.ctx["elongation"], 2),
        "note": "Constant deceleration suggests dry friction; deceleration proportional to speed suggests viscous; to speed squared, air drag.",
    }


def design_trail_step(d: dict[str, Any]) -> dict[str, Any]:
    rec = d["recommended"]
    verdict = (f"The cheapest decisive one: {rec['label']}. The tied equations would then differ by "
               f"{rec['separation_mm']:.0f} mm, {rec['separation_vs_noise']:.1f} times this clip's tracking noise "
               f"({d['noise_mm']:.1f} mm)." if d["decisive"] else
               f"None of them separates the equations clearly; the best is {rec['label']} "
               f"({rec['separation_vs_noise']:.1f} times the tracking noise).")
    return {
        "step": "design", "provider": None, "model": "simulate-and-compare", "status": "accepted" if d["decisive"] else "tied",
        "text": f"Simulated {d['options_tested']} possible experiments with each tied equation. {verdict}",
        "items": [{"model_id": f"option{i}", "title": o["label"][0].upper() + o["label"][1:],
                   "status": "accepted" if o["separation_vs_noise"] >= d["target_ratio"] else "tied",
                   "why": f"predictions differ by {o['separation_mm']:.0f} mm ({o['separation_vs_noise']:.1f} times the noise)"}
                  for i, o in enumerate(d["top"])],
    }


def analyze_bounces(full: Trajectory, origin_px: np.ndarray, ppm: float) -> dict[str, Any] | None:
    """Hybrid bounce model on the whole track; None when there aren't two clean flights to compare."""
    try:
        b = fit_bounces(full)
    except BounceError as e:
        log.info("No bounce analysis: %s", e)
        return None
    pts = b.predicted_xy
    ghost_points = [[int(f), round(float(origin_px[0] + x * ppm), 2), round(float(origin_px[1] - y * ppm), 2)]
                    for f, (x, y) in zip(full.frames, pts, strict=True)]
    return {
        "count": len(b.restitution), "g": b.g, "g_std": b.g_std, "e_mean": b.e_mean, "e_std": b.e_std,
        "impacts": [{"t": t, "e": e, "e_std": es, "tangential": tg}
                    for t, e, es, tg in zip(b.impact_times, b.restitution, b.restitution_std, b.tangential, strict=True)],
        "floor_y_m": b.floor_y, "settled_at": b.settled_at, "rms_m": b.rms_m, "forecast_nrmse": b.forecast_nrmse,
        "accepted": b.accepted, "latex": BOUNCE_LATEX, "ghost": ghost_points,
    }


def bounce_trail_step(b: dict[str, Any]) -> dict[str, Any]:
    settled = f" After {b['settled_at']:.1f} s the hops were too small to resolve." if b["settled_at"] else ""
    verdict = (f"Using e from the first bounces, the model predicts the later ones within {b['forecast_nrmse']:.1%}."
               if b["accepted"] else f"It does not predict the later bounces well ({b['forecast_nrmse']:.0%} error).")
    return {
        "step": "collision", "provider": None, "model": "simulate-and-compare", "status": "accepted" if b["accepted"] else "ambiguous",
        "text": (f"Found {b['count']} bounces. Between impacts the ball is in free fall with g = {b['g']:.3f} ± {b['g_std']:.3f} m/s² "
                 f"(one g shared by every flight); each impact keeps e = {b['e_mean']:.3f} ± {b['e_std']:.3f} of the vertical speed. "
                 f"{verdict}{settled}"),
        "items": [{"model_id": f"impact{i}", "title": f"Bounce {i + 1} at {imp['t']:.2f} s",
                   "status": "accepted" if b["accepted"] else "tied",
                   "why": f"e = {imp['e']:.3f} ± {imp['e_std']:.3f}, horizontal speed kept {imp['tangential']:.0%}"}
                  for i, imp in enumerate(b["impacts"])],
    }


def find_regimes(traj: Trajectory, family: str, candidate_ids: list[str], origin_px: np.ndarray,
                 ppm: float) -> tuple[RegimeChange | None, list[dict[str, Any]]]:
    """Detect a change in dynamics and fit each side separately (e.g. a string shortened mid-clip)."""
    # The richest candidate, so ordinary damping or drag isn't mistaken for a change.
    detector = max(candidate_ids, key=lambda m: len(MODELS[m].params))
    change = detect_change(traj, family, detector)
    if change is None:
        return None, []
    # A few samples from the wrong side would bend a circle fit badly, so each side keeps only certain data.
    regimes = []
    for part in (traj.slice(0, change.pre_end), traj.slice(change.index, len(traj))):
        if len(part) < 2 * traj.fps:
            continue
        coords, v = fit_and_verify(part, family, candidate_ids)
        lead = v.winner or (v.contenders[0] if v.contenders else None)
        if coords is None or lead is None:
            regimes.append({"start_s": float(part.t[0]), "end_s": float(part.t[-1]), "status": v.status,
                            "reason": v.reason, "model_id": None, "title": None, "params": {}, "geometry": {}, "ghost": []})
            continue
        regimes.append({
            "start_s": round(float(part.t[0]), 2), "end_s": round(float(part.t[-1]), 2), "status": v.status,
            "reason": v.reason, "model_id": lead.model_id, "title": MODELS[lead.model_id].title,
            "params": {k: round(x, 4) for k, x in lead.params.items()},
            "geometry": {k: round(x, 4) for k, x in coords.ctx.items() if k in ("length_m", "length_std")},
            "ghost": ghost(coords, lead, part, origin_px, ppm)["points"],
        })
    return change, regimes


def regime_trail_step(change: RegimeChange, regimes: list[dict[str, Any]]) -> dict[str, Any]:
    def describe(r: dict[str, Any]) -> str:
        bits = [f"{k} = {v:.3g}" for k, v in r["params"].items()]
        if "length_m" in r["geometry"]:
            bits.append(f"L = {r['geometry']['length_m']:.3g} m")
        return f"{r['start_s']:.1f} to {r['end_s']:.1f} s: " + (", ".join(bits) if bits else r["reason"])

    return {
        "step": "regime", "provider": None, "model": "simulate-and-compare",
        "text": (f"The dynamics changed at about t = {change.t_estimate:.1f} s: the model fitted before then stops "
                 f"predicting the motion (error {change.error_ratio:.0f}x the normal level). Each part was fitted separately."),
        "items": [{"model_id": r["model_id"] or f"regime{i}", "title": r["title"] or f"Regime {i + 1}",
                   "status": {"accepted": "accepted", "ambiguous": "tied"}.get(r["status"], "rejected"), "why": describe(r)}
                  for i, r in enumerate(regimes)],
    }


def track_objects(client: CostGuardClient, clip: Path, info: media.VideoInfo, scene: Scene, run_id: str,
                  audience: str) -> dict[str, ObjectTrack]:
    """Moving object (id 1) and reference (id 2) in ONE box-prompt request: half the SAM cost of two concept prompts."""
    objects = [msp.PromptObject(id=int(MOVING_ID), prompts=[msp.FramePrompt(frame=0, box=_box(scene.moving_object_box, info))])]
    if scene.reference_box is not None:
        objects.append(msp.PromptObject(id=int(REFERENCE_ID), prompts=[msp.FramePrompt(frame=0, box=_box(scene.reference_box, info))]))
    if scene.second_object_box is not None:
        objects.append(msp.PromptObject(id=int(SECOND_ID), prompts=[msp.FramePrompt(frame=0, box=_box(scene.second_object_box, info))]))
    prompt = msp.build_object_prompt(size=(info.width, info.height), objects=objects)
    r = client.sam_video(prompt=prompt, video_path=clip, video_hash=file_digest(clip), est_frames=info.frames,
                         metadata=SAM_METADATA, run_id=run_id, audience=audience)
    try:
        return tracks_from_text(r.text, completed=r.finish_reason in (None, "completed"))
    except TrackingError as e:
        raise AnalysisError(str(e)) from e


def _box(box, info: media.VideoInfo) -> msp.PromptBox:
    x1, y1, x2, y2 = box.to_pixels(info.width, info.height)
    return msp.PromptBox(x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))


def calibrate(scene: Scene, reference: ObjectTrack | None, moving: ObjectTrack, options: RunOptions) -> Calibration:
    if reference is not None and scene.reference_kind != "none":
        try:
            return from_reference(reference, scene.reference_kind)
        except CalibrationError as e:
            if options.pendulum_length_m is None:
                raise AnalysisError(str(e)) from e
            log.warning("Reference calibration failed (%s); using the given pendulum length.", e)
    if options.pendulum_length_m is not None:
        radius_px = fit_circle(moving.centers_px * np.array([1.0, -1.0]))[2]
        return from_known_length(radius_px, options.pendulum_length_m)
    # No scale: carry on. Frequency, damping and friction don't depend on scale; everything measured in
    # meters is reported as uncalibrated instead of the whole analysis being refused.
    log.info("No scale reference; continuing with a nominal scale (lengths and g will be marked uncalibrated).")
    return Calibration(NOMINAL_PX_PER_M, 0.0, "none")


def build_trajectory(moving: ObjectTrack, fps: float, px_per_m: float) -> tuple[Trajectory, np.ndarray]:
    """The collision-free stretch of the track with the most motion."""
    traj, origin = full_track(moving, fps, px_per_m)
    return longest_flight(traj, origin, px_per_m)


def full_track(moving: ObjectTrack, fps: float, px_per_m: float) -> tuple[Trajectory, np.ndarray]:
    segments = fill_short_gaps(moving.frames, moving.centers_px, max_gap=3)
    frames, centers = max(segments, key=lambda s: len(s[0]))
    return from_pixels(frames, centers, fps, px_per_m), centers[0]


def longest_flight(traj: Trajectory, origin: np.ndarray, px_per_m: float) -> tuple[Trajectory, np.ndarray]:
    frames = traj.frames
    centers = origin + np.column_stack([traj.xy[:, 0], -traj.xy[:, 1]]) * px_per_m
    # Generic jumps plus floor bounces: a missed small bounce would merge two flights into one "free fall".
    impacts = np.union1d(detect_impacts(traj), detect_bounces(traj))
    pieces = split_at(traj, impacts)
    if pieces:
        # Most motion, not most samples: after the last bounce a resting ball makes a long, flat, useless piece.
        best = max(pieces, key=lambda p: float(np.ptp(p.xy, axis=0).max()))
        start = int(np.searchsorted(frames, best.frames[0]))
        return from_pixels(best.frames, centers[start:start + len(best)], traj.fps, px_per_m), centers[start]
    return traj, centers[0]


def fit_and_verify(traj: Trajectory, family: str, candidate_ids: list[str],
                   coords: Coordinates | None = None) -> tuple[Coordinates | None, Verdict]:
    if family == "rotation" and coords is None:
        return None, Verdict("no_fit", None, (), (), "The spin was not measured (the scene was not read as a rotation).")
    try:
        coords = coords if coords is not None else to_coordinates(traj, family)
    except ValueError as e:
        return None, Verdict("no_fit", None, (), (), f"The motion does not look like a {family}: {e}")
    fits = [fit_model(MODELS[m], coords) for m in candidate_ids]
    return coords, assess(coords, fits)


def _hypothesis_items(hyp) -> list[dict[str, Any]]:
    return [{"model_id": h.model_id, "title": MODELS[h.model_id].title, "why": h.rationale} for h in hyp.candidates]


def _with_control(family: str, ids: list[str]) -> list[str]:
    """Always include the family's simplest model, so a hypothesis list can't hide the baseline."""
    simplest = min(models_for(family), key=lambda m: len(m.params)).id
    return list(dict.fromkeys([simplest, *ids]))


def agent_context(v: Verdict, family: str, summary: dict[str, Any], scene: Scene, tried: list[str]) -> dict[str, Any]:
    return {
        "verdict": v.status, "verdict_reason": v.reason, "family": family, "families_tried": tried,
        "object": scene.moving_object,
        "candidates": [fit_summary(f) for f in v.ranked],
        "remaining_contenders": [f.model_id for f in v.contenders],
        "trajectory_summary": summary,
    }


def fit_summary(f: FitResult, rationale: str | None = None) -> dict[str, Any]:
    return {"model_id": f.model_id, "title": MODELS[f.model_id].title, "params": f.params, "bic": f.bic,
            "heldout_nrmse": f.heldout_nrmse, "rmse": f.rmse, "rationale": rationale}


def derived_parameters(c: Coordinates) -> list[dict[str, Any]]:
    if c.family == "pendulum":
        L, s = c.ctx["length_m"], c.ctx["length_std"]
        return [{"name": "L", "latex": "L", "unit": "m", "value": L, "std": s, "ci95": [L - 1.96 * s, L + 1.96 * s],
                 "derived": "radius of the fitted swing arc"}]
    return []


def _track_points(frames: np.ndarray, xy: np.ndarray, origin_px: np.ndarray, ppm: float) -> list[list[float]]:
    px = origin_px[0] + xy[:, 0] * ppm
    py = origin_px[1] - xy[:, 1] * ppm
    return [[int(f), round(float(x), 2), round(float(y), 2)] for f, x, y in zip(frames, px, py, strict=True)]


def ghost(c: Coordinates, f: FitResult, traj: Trajectory, origin_px: np.ndarray, ppm: float) -> dict[str, Any]:
    return {"model_id": f.model_id, "points": _track_points(traj.frames, c.to_xy(f.sim), origin_px, ppm)}


def trajectory_from_result(result: dict[str, Any]) -> Trajectory:
    """Rebuild the calibrated trajectory from a stored result (its pixel track, frame rate and scale)."""
    points = np.array(result["real_track_px"], dtype=float)
    if len(points) < 10:
        raise AnalysisError("This analysis has too little track to reuse.")
    return from_pixels(points[:, 0].astype(int), points[:, 1:3], float(result["clip"]["fps"]),
                       float(result["calibration"]["px_per_m"]))
