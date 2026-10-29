"""Recorded mode end to end: video -> scene -> SAM tracks -> calibrated trajectory -> hypotheses -> verified equation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import meta_sam_parser as msp
import numpy as np

from .agent.llm import LLMError
from .agent.reasoning import (
    SampledHypotheses,
    agent_turn,
    check_track,
    locate_objects,
    propose_hypotheses,
    propose_hypotheses_sampled,
    understand_scene,
)
from .agent.schemas import AgentTurn, Scene
from .agent.skeptic import challenge, fit_more, outcome_text
from .costguard import CostGuardClient, CostGuardError
from .costguard.cache import file_digest
from .export.mjcf import MjcfError, calibrate_contact, check_against_fit
from .perception import media
from .perception.calibration import (
    Calibration,
    CalibrationError,
    from_known_length,
    from_reference,
)
from .perception.sam_track import ObjectTrack, TrackingError, tracks_from_text
from .perception.track_quality import assess_track, overlay_frames
from .physics import depth3d
from .physics.audit import AuditInput, LawEvidence, audit
from .physics.audit import cant_judge as audit_cant_judge
from .physics.collisions import BounceError, detect_bounces, fit_bounces
from .physics.coords import Coordinates, fit_circle, to_coordinates
from .physics.custom import FREE_MODELS, FreeParam, build, register, spec_for
from .physics.design import design_experiment
from .physics.double import HORIZON_S as DOUBLE_HORIZON_S
from .physics.double import (
    DoublePendulumError,
    fit_double,
    interior_start,
    to_double_coords,
)
from .physics.double import simulate as double_simulate
from .physics.double import summary as double_summary
from .physics.fit import FitResult, fit_model
from .physics.models import MODELS, models_for, needs_scale
from .physics.regime import RegimeChange, detect_change
from .physics.rotation import RotationError, rotation_coordinates
from .physics.sindy import vet
from .physics.uncertainty import bootstrap
from .physics.verify import Verdict, assess, judge_candidates, window_indices
from .sigproc.cleanup import detect_impacts, differentiate, fill_short_gaps, split_at
from .sigproc.summary import summarize
from .sigproc.trajectory import Trajectory, from_pixels
from .sound import audio_trail_step, listen

log = logging.getLogger("physicslens.pipeline")

REASONING_TRAIL_CHARS = 6000  # long thoughts are cut for the trail; llm_calls keeps the full text
NOMINAL_PX_PER_M = 1000.0  # used only when nothing in the clip gives a scale
SCALE_NOTE = ("No reference of known size was found, so lengths and g are in arbitrary units. Frequency, damping "
              "and friction rates don't depend on scale and are exact. For a calibrated result, put an A4 sheet "
              "flat in the plane of motion or enter the pendulum length.")



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
    object_size_m: float | None = None  # known longest dimension of the moving object (a ball, a car, an astronaut)
    object_hint: str | None = None  # the user's words for what to measure, e.g. "the astronaut"
    slow_motion: float = 1.0  # capture rate / playback rate, e.g. 8 for a phone's 240 fps slow motion saved at 30 fps
    prepared: Path | None = None  # a media.save_prepared bundle: analyze exactly these bytes (see save_prepared)
    blade_count: int | None = None  # fan blades (or other sound sources per turn): turns a blade-pass tone into a spin rate
    depth_3d: bool = False  # also measure a ball's depth from its apparent size (needs object_size_m and focal_px)
    focal_px: float | None = None  # camera focal length in pixels at the uploaded video's resolution
    # A rendered or instrument-calibrated video (orbit mode): the image scale and the fixed point the coordinates are
    # measured from (the Sun) are part of the frame's geometry, given by whoever drew it. The law is still discovered.
    known_px_per_unit: float | None = None
    origin_px: tuple[float, float] | None = None
    units: dict[str, str] | None = None  # display names when 1 m and 1 s stand for other units, e.g. AU and day
    frame_note: str | None = None  # what the coordinates mean, passed to the reasoning model with the numbers


Progress = Callable[[str], None]


def analyze(client: CostGuardClient, video: Path, run_dir: Path, run_id: str, options: RunOptions | None = None,
            progress: Progress = lambda _msg: None) -> dict[str, Any]:
    return finite(_analyze(client, video, run_dir, run_id, options, progress))


def finite(obj: Any) -> Any:
    """NaN or infinity becomes null: JSON has no such numbers, and one would break the API and every browser."""
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [finite(v) for v in obj]
    if isinstance(obj, float | np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _analyze(client: CostGuardClient, video: Path, run_dir: Path, run_id: str, options: RunOptions | None = None,
             progress: Progress = lambda _msg: None) -> dict[str, Any]:
    options = options or RunOptions()
    run_dir.mkdir(parents=True, exist_ok=True)
    limits = client.limits
    media_cfg = limits.get("media") or {}
    max_turns = int((limits.get("agent") or {}).get("max_turns", 4))
    llm_calls: list[dict[str, Any]] = []
    # What the agent and the verifier decided, in order, each tagged with who decided it (shown on screen).
    trail: list[dict[str, Any]] = []

    def record(step: str, call, text: str | None, **extra: Any) -> None:
        """text None: the call is logged in llm_calls but adds no trail line (extra samples of one step)."""
        llm_calls.append({"step": step, "provider": call.provider, "model": call.model, "cached": call.cached,
                          "reasoning": call.reasoning, "fallbacks": list(call.fallbacks)})
        if text is None:
            return
        if call.fallbacks:
            extra = {**extra, "fallbacks": list(call.fallbacks)}
        if call.reasoning:
            # What the model thought before answering; the UI shows it collapsed under the step.
            extra = {**extra, "reasoning": call.reasoning[:REASONING_TRAIL_CHARS]}
        trail.append({"step": step, "provider": call.provider, "model": call.model, "text": text, **extra})

    progress("Preparing video")
    clip, info, frames_jpeg = (media.load_prepared(options.prepared, run_dir) if options.prepared is not None
                               else prepare(video, run_dir, options, media_cfg))
    min_frames = int(media_cfg.get("sam_min_frames_per_request", 50))
    if info.frames < min_frames:
        raise AnalysisError(f"The clip is too short: at least {min_frames} frames (about 2 seconds) are needed.")

    progress("Understanding the scene")
    hint = (options.object_hint or "").strip()[:60] or None
    scene, call = understand_scene(client, frames_jpeg, run_id, options.audience, hint=hint)
    ref = "no reference object" if scene.reference_kind == "none" else f"reference: {scene.reference_kind.replace('_', ' ')}"
    told = f" (asked to measure: {hint})" if hint else ""
    record("scene", call, f"Sees {article(scene.moving_object)} {scene.moving_object} {MOTION_PHRASE[scene.motion_family]}; "
                          f"{ref}{told}.")

    scene, prompt_frame = locate_precisely(client, scene, frames_jpeg, info, run_id, options.audience, record, limits, progress)

    progress(f"Tracking the {scene.moving_object} with SAM 3.1")
    tracks = track_objects(client, clip, info, scene, run_id, options.audience, prompt_frame)
    moving = tracks.get(MOVING_ID)
    if moving is None or len(moving.frames) < min_frames // 2:
        raise AnalysisError(f"SAM could not follow the {scene.moving_object} through the clip.")
    trail.append({"step": "tracking", "provider": "meta_api", "model": "sam-3.1",
                  "text": f"Tracked the {scene.moving_object} in {len(moving.frames)} of {info.frames} frames."})
    moving = check_and_repair_track(client, clip, info, scene, moving, run_id, options.audience, record, trail, progress)
    tracks[MOVING_ID] = moving
    if scene.motion_family != "rotation":  # a spin turns in place; everything else must actually travel
        travel = float(np.ptp(moving.centers_px, axis=0).max())
        size = float(np.median(moving.long_px))
        # Generous on purpose: a jumping person rises a fifth of their height, a chalk mark doesn't move at all.
        moves_a_lot = travel >= max(0.25 * size, 0.02 * max(info.width, info.height))
        # A big object can move a little and still be measured: a truck's body bounced 6 px on its springs with
        # 0.1 to 0.2 px of frame-to-frame jitter (ratio 34 to 58), while a stopwatch rim SAM locked onto by mistake
        # "moved" 5 px with 0.5 px of jitter (ratio 11). Smooth motion well above the jitter is real motion.
        if not moves_a_lot and not smooth_motion(moving.centers_px, travel):
            # Fitting equations to a mark that never moved would only find patterns in noise.
            raise AnalysisError(
                f"What SAM followed as the {scene.moving_object} barely moves ({travel:.0f} px). It may have locked on to "
                "the wrong thing: say what to measure in the hint field, or start the clip where the object is in view.")

    progress("Calibrating scale")
    calibration = calibrate(scene, tracks.get(REFERENCE_ID), moving, options)
    # Physics runs in real time; drawing on the clip uses frame indices, so the video keeps its own rate.
    fps = info.fps * options.slow_motion
    full, full_origin = full_track(moving, fps, calibration.px_per_m, options.origin_px)
    # The whole track until the family is known: splitting at "impacts" only makes sense for a thrown or
    # bouncing object, and would chop an oscillator at its turning points (found on a real spring clip).
    traj, origin_px = full, full_origin
    rotation: Coordinates | None = None
    if scene.motion_family == "rotation":
        progress("Measuring the spin from the mask orientation")
        rotation, traj, origin_px = rotation_track(moving, fps, calibration.px_per_m)

    summary = rotation_summary(rotation) if rotation is not None else summarize(traj, calibration.source == "none")
    if options.frame_note:
        # Every reasoning step that reads the numbers also reads what they mean (units, where the origin is).
        summary = {**summary, "frame_note": options.frame_note}
    n_samples = int((limits.get("agent") or {}).get("hypothesis_samples", 1))
    progress("Proposing hypotheses" + (f" ({n_samples} independent proposals)" if n_samples > 1 else ""))
    sampled = propose_hypotheses_sampled(client, summary, scene, run_id, options.audience, samples=n_samples)
    hyp = sampled.merged
    record("hypotheses", sampled.calls[0], _sampled_text(sampled), items=_hypothesis_items(hyp, sampled))
    for extra_call in sampled.calls[1:]:
        record("hypotheses", extra_call, None)

    if hyp.family == "double_pendulum":
        second = tracks.get(SECOND_ID)
        if second is None:
            raise AnalysisError("A double pendulum needs both bobs tracked, and SAM did not find the outer one.")
        progress("Fitting the double pendulum with short-horizon predictions")
        dbl, ghosts, steps = double_analysis(moving, second, fps, calibration.px_per_m)
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
            "clip": {"width": info.width, "height": info.height, "fps": info.fps, "frames": info.frames,
                 "slow_motion": options.slow_motion},
            "calibration": {"px_per_m": calibration.px_per_m, "rel_std": calibration.rel_std, "source": calibration.source},
            "agent": {"action": turn.action, "message": turn.message, "reasoning_summary": turn.reasoning_summary},
            "ranking": [], "real_track_px": dbl.pop("inner_track_px"), "object_size_px": float(np.median(moving.long_px)),
            "llm_calls": llm_calls, "trail": trail, "summary": summary, "ghosts": ghosts, "double": dbl,
            **({"scale_note": SCALE_NOTE} if calibration.source == "none" else {}),
            "audit": audit_cant_judge("the physics audit doesn't cover double pendulums yet: chaos amplifies tracking "
                                      "noise, so slow drifts can't be told apart from it."),
        }

    tried: list[str] = []
    family = hyp.family
    full_clip_verdict: dict[str, Any] | None = None  # set when the agent's fit window was used
    candidate_ids, rationales = _candidates(hyp)
    verdict: Verdict | None = None
    coords: Coordinates | None = None
    turn = None
    change: RegimeChange | None = None
    regimes: list[dict[str, Any]] = []
    bounces: dict[str, Any] | None = None
    agent_cfg = limits.get("agent") or {}
    tool_steps_left = int(agent_cfg.get("max_tool_steps", 4))
    tool_deadline = time.monotonic() + float(agent_cfg.get("max_tool_seconds", 120))
    notebook: list[dict[str, Any]] = []
    for _ in range(max_turns):
        tried.append(family)
        if rotation is None:
            traj, origin_px = (longest_flight(full, full_origin, calibration.px_per_m) if family == "projectile"
                               else (full, full_origin))
        unwindowed = traj
        if rotation is None and hyp.fit_window is not None and family == hyp.family and len(tried) == 1:
            w = hyp.fit_window
            span = window_indices(traj.t, w.start_s, w.end_s)
            if isinstance(span, str):
                trail.append({"step": "plausibility", "provider": None, "model": "simulate-and-compare", "status": "rejected",
                              "text": f"Refused the agent's fit window ({w.start_s:.2f} to {w.end_s:.2f} s): {span}. Fitting the whole clip."})
            else:
                traj = traj.slice(*span)
                trail.append({"step": "plausibility", "provider": None, "model": "simulate-and-compare", "status": "accepted",
                              "text": f"Fits {w.start_s:.2f} to {w.end_s:.2f} s as the agent proposed ({w.reason}); the "
                                      "whole clip is fitted too and its verdict kept beside this one."})
        progress(f"Fitting and simulating {len(candidate_ids)} {family} models")
        given = known_length_geometry(moving, origin_px, calibration, options) if family == "pendulum" else None
        coords, verdict = fit_and_verify(traj, family, candidate_ids,
                                         rotation if family == "rotation" else
                                         to_coordinates(traj, "pendulum", given) if given else None)
        if unwindowed is not traj:
            _, whole = fit_and_verify(unwindowed, family, candidate_ids,
                                      to_coordinates(unwindowed, "pendulum", given) if given else None)
            full_clip_verdict = {"status": whole.status, "reason": whole.reason,
                                 "winner": whole.winner.model_id if whole.winner else None}
        judged = judge_candidates(verdict)
        trail.append({"step": "verification", "provider": None, "model": "simulate-and-compare", "text": verdict.reason,
                      "status": verdict.status,
                      "items": [{"model_id": m, "title": MODELS[m].title, "status": st, "why": why} for m, (st, why) in judged.items()]})
        if family == "projectile" and bounces is None:
            progress("Analyzing bounces")
            bounces = analyze_bounces(full, full_origin, calibration.px_per_m)
            if bounces is not None and calibration.rel_std > 0:
                bounces["g_std"] = float(np.hypot(bounces["g_std"], bounces["g"] * calibration.rel_std))
            if bounces is not None and bounces.get("accepted"):
                # The measured restitution as MuJoCo contact parameters, tuned inside MuJoCo until its bounce matches.
                radius = float(np.median(moving.long_px)) / calibration.px_per_m / 2
                contact = calibrate_contact(bounces["g"], bounces["e_mean"], max(radius, 0.005))
                bounces["mujoco_contact"] = {k: contact[k] for k in ("dampratio", "timeconst", "e_mujoco")} | {"radius_m": radius}
            if bounces is not None:
                trail.append(bounce_trail_step(bounces))
        design = None
        if verdict.status == "ambiguous" and coords is not None and len(verdict.contenders) >= 2:
            progress("Designing the experiment that would separate the tied models")
            design = design_experiment(coords, list(verdict.contenders))
            if design is not None:
                trail.append(design_trail_step(design, calibration.source == "none"))
        progress("Checking whether the dynamics change during the clip")
        change, regimes = (None, []) if family == "rotation" else find_regimes(traj, family, candidate_ids, origin_px, calibration.px_per_m)
        if change is not None:
            trail.append(regime_trail_step(change, regimes))
        progress("Agent reviewing the verdict")
        context = agent_context(verdict, family, summary, scene, tried)
        if calibration.source == "none":
            context["scale"] = ("none, so the app will use Earth's gravity as the ruler: lengths will be reported in meters "
                                "assuming g = 9.81 m/s^2; say so if you mention a length"
                                if gravity_ruler_applies(verdict.winner, family) else
                                "none: lengths and g are uncalibrated, frequency, damping and friction are exact")
        if design is not None:
            context["experiment_design"] = {k: design[k] for k in ("recommended", "decisive", "noise_mm", "models")}
        if bounces is not None:
            context["bounces"] = {k: bounces[k] for k in ("g", "g_std", "e_mean", "e_std", "count", "accepted", "forecast_nrmse")}
        if change is not None:
            context["regime_change"] = {"at_s": round(change.t_estimate, 2), "regimes": [
                {k: r[k] for k in ("start_s", "end_s", "status", "title", "params")} for r in regimes]}
        # The agent may test more models or inspect part of the motion before deciding; code recomputes the verdict.
        while True:
            context.update(agent_context(verdict, family, summary, scene, tried))
            context["tool_steps_left"] = tool_steps_left if coords is not None else 0
            context["notebook"] = notebook
            context["untested_models"] = [{"model_id": m.id, "title": m.title} for m in models_for(family)
                                          if m.id not in {f.model_id for f in verdict.ranked} and not m.id.startswith("custom_")]
            turn, call = agent_turn(client, context, run_id, options.audience)
            if turn.action not in TOOL_ACTIONS:
                break
            if tool_steps_left <= 0 or coords is None or time.monotonic() > tool_deadline:
                record("agent", call, "Wanted to test more, but its step budget is used up; the verdict stands.",
                       action=turn.action, status="rejected", reasoning_summary=turn.reasoning_summary)
                turn = AgentTurn(action=FINAL_FOR[verdict.status], message=verdict.reason, next_family=None,
                                 reasoning_summary="Step budget used up; the verdict from numerical verification stands.")
                break
            tool_steps_left -= 1
            if turn.action == "inspect":
                note = inspect_window(traj, turn, calibration.source == "none")
                record("agent", call, f"{turn.message} {note['text']}", action="inspect",
                       reasoning_summary=turn.reasoning_summary, expectation=turn.expectation)
                notebook.append(note["entry"])
                continue
            before = verdict
            proposals = [(m, None, turn.expectation or turn.message) for m in turn.model_ids]
            if turn.free_model is not None:
                proposals.append((None, turn.free_model, turn.expectation or turn.message))
            verdict, items, more = fit_more(coords, verdict, family, proposals, source="the agent")
            rationales.update(more)
            candidate_ids = list(dict.fromkeys([*candidate_ids, *(i["model_id"] for i in items if i["status"] != "untestable")]))
            outcome = tool_outcome(before, verdict)
            record("agent", call, f"{turn.message} {outcome}", action="test_models", items=items,
                   status=verdict.status, reasoning_summary=turn.reasoning_summary, expectation=turn.expectation)
            notebook.append({"tested": [i["model_id"] for i in items], "results": [
                {k: i.get(k) for k in ("model_id", "status", "why", "heldout_nrmse")} for i in items],
                "verdict_after": verdict.status, "winner_after": verdict.winner.model_id if verdict.winner else None})
            if verdict.status == "ambiguous" and verdict is not before and len(verdict.contenders) >= 2:
                design = design_experiment(coords, list(verdict.contenders))
                if design is not None:
                    trail.append(design_trail_step(design, calibration.source == "none"))
                    context["experiment_design"] = {k: design[k] for k in ("recommended", "decisive", "noise_mm", "models")}
        if turn.action != FINAL_FOR[verdict.status] and not (verdict.status == "no_fit" and turn.action in ("try_family", "give_up")):
            # Wording is the model's; the verdict is not. A mismatch is shown, never acted on.
            trail.append({"step": "verification", "provider": None, "model": "simulate-and-compare", "status": verdict.status,
                          "text": f"The agent chose \"{turn.action}\", which does not match the verdict ({verdict.status}); the verdict stands."})
        record("agent", call, turn.message, action=turn.action, reasoning_summary=turn.reasoning_summary)
        if verdict.status != "no_fit" or turn.action != "try_family" or not turn.next_family or turn.next_family in tried:
            break
        hyp, call = propose_hypotheses(client, summary, scene, run_id, options.audience, exclude_families=tried)
        record("hypotheses", call, f"Switches to {hyp.family} models.", items=_hypothesis_items(hyp))
        if hyp.family in tried:
            break
        family = hyp.family
        candidate_ids, more = _candidates(hyp)
        rationales.update(more)

    assert verdict is not None and turn is not None
    skeptic: dict[str, Any] | None = None
    if verdict.status == "accepted" and verdict.winner is not None and coords is not None:
        progress("A referee is trying to break the accepted law")
        winner_title = MODELS[verdict.winner.model_id].title
        try:
            ch = challenge(client, coords, verdict, family, summary, scene, run_id, options.audience)
        except (LLMError, CostGuardError) as e:
            # The accepted law stands on its own verification; a referee that could not run is reported, not hidden.
            log.warning("skeptic skipped: %s", e)
            skeptic = {"skipped": str(e)}
            trail.append({"step": "skeptic", "provider": None, "model": "simulate-and-compare", "status": "skipped",
                          "text": "The referee could not run this time, so the law was not challenged."})
        else:
            skeptic = ch.info
            record("skeptic", ch.call, outcome_text(ch.info, winner_title), status="accepted" if ch.info["survived"] else "rejected",
                   items=ch.info["challenges"], reasoning_summary=ch.review.reasoning_summary)
            if ch.info["concerns"]:
                trail.append({"step": "skeptic", "provider": None, "model": "simulate-and-compare", "status": "untested",
                              "text": "Raised but not testable from this clip: " + " ".join(ch.info["concerns"])})
            if not ch.info["survived"]:
                verdict = ch.verdict
                rationales.update(ch.rationales)
                context = agent_context(verdict, family, summary, scene, tried)
                context["referee"] = outcome_text(ch.info, winner_title)
                turn, call = agent_turn(client, context, run_id, options.audience)
                record("agent", call, turn.message, action=turn.action, reasoning_summary=turn.reasoning_summary)
    result: dict[str, Any] = {
        "status": verdict.status,
        "reason": verdict.reason,
        "family": family,
        "families_tried": tried,
        "scene": scene.model_dump(),
        "clip": {"width": info.width, "height": info.height, "fps": info.fps, "frames": info.frames,
                 "slow_motion": options.slow_motion},
        "calibration": {"px_per_m": calibration.px_per_m, "rel_std": calibration.rel_std, "source": calibration.source},
        "agent": {"action": turn.action, "message": turn.message, "reasoning_summary": turn.reasoning_summary},
        "ranking": [fit_summary(f, rationales.get(f.model_id)) for f in verdict.ranked],
        "real_track_px": _track_points(traj.frames, traj.xy, origin_px, calibration.px_per_m),
        # For a spin the ghost is a small marker on the rim, not the whole object.
        "object_size_px": float(np.median(moving.long_px)) * (0.25 if rotation is not None else 1.0),
        "llm_calls": llm_calls,
        "full_clip_verdict": full_clip_verdict,
        "skeptic": skeptic,
        "hypothesis_votes": _winner_votes(sampled, verdict),
        "units": options.units,
        "trail": trail,
        "summary": summary,
    }
    if coords is not None:
        try:
            vetted = vet(coords, verdict.winner if verdict.status == "accepted" else None)
            result["sindy"] = vetted["equations"] if vetted["shown"] else None
            result["sindy_check"] = vetted
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
    if lead is not None and coords is not None and lead.sim_train.size:
        result["forecast"] = forecast_series(coords, lead)
    if lead is not None and coords is not None:
        # Everything a what-if needs to re-simulate from the measured starting state and draw it on the clip.
        result["fit_state"] = {
            "model_id": lead.model_id, "params": lead.params, "y0": [float(v) for v in lead.y0],
            "ctx": {k: float(v) for k, v in coords.ctx.items()}, "fps": traj.fps,
            "origin_px": [float(origin_px[0]), float(origin_px[1])], "px_per_m": calibration.px_per_m,
            "first_frame": int(traj.frames[0]),
        }
        if lead.model_id in FREE_MODELS:
            result["fit_state"]["free_model"] = FREE_MODELS[lead.model_id].definition()
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
    g_fit = next((p for p in result.get("parameters", []) if p["name"] == "g"), None)
    if (calibration.source == "none" and gravity_ruler_applies(verdict.winner, family) and g_fit is not None
            and g_fit["std"] < RULER_MAX_REL_STD * g_fit["value"]):
        calibration = apply_gravity_ruler(result)
        if "fit_state" in result:
            try:
                result["mujoco"] = check_against_fit(result["fit_state"], len(traj) / traj.fps)
            except MjcfError as e:
                log.warning("MuJoCo check skipped after rescaling: %s", e)
    add_scale_error(result.get("parameters", []), calibration.rel_std)
    if calibration.source == "none":
        result["scale_note"] = SCALE_NOTE
        for p in result.get("parameters", []):
            p["uncalibrated"] = needs_scale(p["unit"])
    progress("Listening to the soundtrack")
    result["audio"] = listen(video, run_dir, info.frames / info.fps, options.slow_motion, family, result, options.blade_count,
                             rotation, traj.frames if rotation is not None else None, info.fps)
    if (step := audio_trail_step(result["audio"])) is not None:
        trail.append(step)
    progress("Auditing the motion for physics violations")
    result["audit"] = audit_clip(full, family, calibration, verdict, moving, tracks.get(REFERENCE_ID), info, change, result, video)
    if options.depth_3d:
        progress("Measuring the ball's depth from its apparent size")
        result["depth3d"] = depth_for_run(moving, traj, fps, info, video, options, scene.moving_object, family)
    progress("Done")
    return result


def audit_clip(full: Trajectory, family: str, calibration: Calibration, verdict: Verdict, moving: ObjectTrack,
               reference: ObjectTrack | None, info: media.VideoInfo, change: RegimeChange | None,
               result: dict[str, Any], video: Path) -> dict[str, Any]:
    """The physics audit on the whole track, with what the rest of the analysis found. Local math only."""
    real_scale = calibration.source not in ("none", "gravity")
    g = next((p for p in result.get("parameters", []) if p["name"] == "g"), None) if real_scale else None
    b = result.get("bounces") if real_scale else None
    if g is None and b and b.get("accepted"):
        g = {"value": b["g"], "std": b["g_std"]}
    best = verdict.winner or (verdict.ranked[0] if verdict.ranked else None)
    law = LawEvidence(verdict.status, best.model_id if best else None, best.heldout_nrmse if best else None,
                      best.heldout_rmse / best.noise_sigma if best and best.noise_sigma > 0 else None,
                      g["value"] if g else None, g["std"] if g else None)
    # Mask size per sample of the track (its fitted rectangle), interpolated over the short gaps the track fills.
    areas = np.interp(full.frames, moving.frames, moving.long_px * moving.short_px)
    try:
        varies = media.frame_rate_varies(video) if video.is_file() else None
    except media.MediaError as e:
        log.warning("Frame-rate check skipped: %s", e)
        varies = None
    try:
        return audit(AuditInput(
            traj=full, family=family, calibration_source=calibration.source,
            px_per_m=NOMINAL_PX_PER_M if calibration.source == "gravity" else calibration.px_per_m,
            scale_rel_std=calibration.rel_std, law=law, object_size_px=float(np.median(moving.long_px)), areas_px=areas,
            confidence=moving.confidence, coverage=len(moving.frames) / max(info.frames, 1), frame_rate_varies=varies,
            reference_px=reference.centers_px if reference is not None else None,
            reference_size_px=float(np.median(reference.long_px)) if reference is not None else None,
            regime_change_s=change.t_estimate if change is not None else None))
    except (ValueError, np.linalg.LinAlgError) as e:  # the audit is a second opinion; it must not sink the result
        log.warning("Physics audit failed: %s", e)
        return audit_cant_judge(f"the audit's own measurement failed on this track ({e}).")


def depth_for_run(moving: ObjectTrack, traj: Trajectory, fps: float, info: media.VideoInfo, video: Path,
                  options: RunOptions, moving_object: str, family: str) -> dict[str, Any]:
    focal = options.focal_px
    if focal is not None and options.prepared is None:
        # The focal length is measured on the phone's own frames; the clip SAM tracked may be downscaled. A prepared
        # bundle has no original beside it, so there the focal length is taken as given for the stored clip.
        source = media.probe(video)
        focal *= max(info.width, info.height) / max(source.width, source.height)
    return depth3d.for_run(moving, traj.frames, fps=fps, width=info.width, height=info.height, focal_px=focal,
                           size_m=options.object_size_m, moving_object=moving_object, family=family)


def prepare(video: Path, run_dir: Path, options: RunOptions, media_cfg: dict[str, Any], start_s: float = 0.0,
            seconds: float | None = None) -> tuple[Path, media.VideoInfo, list[bytes]]:
    """The clip SAM sees (downsampled, trimmed, H.264) and the few small keyframes the reasoning model sees."""
    run_dir.mkdir(parents=True, exist_ok=True)
    clip = run_dir / "clip.mp4"
    cap = MAX_CLIP_SECONDS[options.audience]
    info = media.prepare_sam_clip(video, clip, max_fps=float(media_cfg.get("sam_max_fps_recorded", 30)),
                                  max_seconds=min(seconds, cap) if seconds else cap, start_s=start_s,
                                  max_long_side=int(media_cfg.get("dev_max_resolution", 1080)))
    frames = media.keyframes(clip, int(media_cfg.get("llm_keyframes_max", 4)), int(media_cfg.get("llm_image_long_side_px", 768)))
    return clip, info, frames


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

    def px(xy: np.ndarray, start: int = 0) -> list[list[float]]:
        return [[int(f), round(float(origin[0] + x * ppm), 2), round(float(origin[1] - y * ppm), 2)]
                for f, (x, y) in zip(frames[start:], xy, strict=True)]

    kin = differentiate(c.q, fps)
    states = np.column_stack([kin.pos, kin.vel])
    anchored = reanchored_prediction(fit.params, c, states, every_s=DOUBLE_HORIZON_S)
    k = interior_start(fps)  # not the very first sample: its smoothed velocity is an edge estimate
    free = double_simulate(fit.params, c.L1, c.L2, states[k], t[k:])
    a_inner, a_outer = c.to_xy(anchored)
    every = f"restarted every {DOUBLE_HORIZON_S:.1f} s"
    ghosts = [{"model_id": f"inner bob (model, {every})", "points": px(a_inner)},
              {"model_id": f"outer bob (model, {every})", "points": px(a_outer)}]
    if free is not None:
        ghosts.append({"model_id": "outer bob (one prediction from the start)", "points": px(c.to_xy(free[:, :2])[1], k)})
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


FORECAST_POINTS = 400
def article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


MOTION_PHRASE = {
    "pendulum": "swinging like a pendulum", "oscillator": "oscillating back and forth",
    "projectile": "flying or falling freely", "double_pendulum": "swinging as a double pendulum",
    "rotation": "spinning in place", "other": "moving in a way none of the named models covers",
}


def forecast_series(c: Coordinates, fit: FitResult) -> dict[str, Any]:
    """Measured motion against the model fitted on the first part only: the verification, as data to plot."""
    spec = MODELS[fit.model_id]
    # One plotted coordinate: the angle for swings and spins, the main axis for springs, height for projectiles.
    k = 1 if c.family in ("projectile", "plane") else 0
    label = {"pendulum": ("Swing angle", "rad"), "oscillator": ("Displacement", "m"), "projectile": ("Height", "m"),
             "rotation": ("Spin angle", "rad"), "line": ("Position along the path", "m"),
             "plane": ("Height", "m")}.get(c.family, (c.names[k], ""))
    step = max(1, len(c.t) // FORECAST_POINTS)
    idx = np.arange(0, len(c.t), step)
    return {
        "model_id": fit.model_id, "title": spec.title, "label": label[0], "unit": label[1],
        "t": [round(float(v), 4) for v in c.t[idx]],
        "measured": [round(float(v), 5) for v in c.q[idx, k]],
        "forecast": [round(float(v), 5) for v in fit.sim_train[idx, k]],
        "heldout_start": round(float(c.t[min(fit.n_train, len(c.t) - 1)]), 4),
        "heldout_nrmse": fit.heldout_nrmse,
    }


def design_trail_step(d: dict[str, Any], uncalibrated: bool = False) -> dict[str, Any]:
    # With no scale the nominal 1000 px/m makes a "mm" a pixel, so say pixels rather than invent millimetres.
    mm = "px" if uncalibrated else "mm"
    rec = d["recommended"]
    verdict = (f"The cheapest decisive one: {rec['label']}. The tied equations would then differ by "
               f"{rec['separation_mm']:.0f} {mm}, {rec['separation_vs_noise']:.1f} times this clip's tracking noise "
               f"({d['noise_mm']:.1f} {mm})." if d["decisive"] else
               f"None of them separates the equations clearly; the best is {rec['label']} "
               f"({rec['separation_vs_noise']:.1f} times the tracking noise).")
    return {
        "step": "design", "provider": None, "model": "simulate-and-compare", "status": "accepted" if d["decisive"] else "tied",
        "text": f"Simulated {d['options_tested']} possible experiments with each tied equation. {verdict}",
        "items": [{"model_id": f"option{i}", "title": o["label"][0].upper() + o["label"][1:],
                   "status": "accepted" if o["separation_vs_noise"] >= d["target_ratio"] else "tied",
                   "why": f"predictions differ by {o['separation_mm']:.0f} {mm} ({o['separation_vs_noise']:.1f} times the noise)"}
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


MIN_TRAVEL_TO_JITTER = 20.0
MIN_SMOOTH_TRAVEL_PX = 3.0


def smooth_motion(centers_px: np.ndarray, travel: float) -> bool:
    if len(centers_px) < 8 or travel < MIN_SMOOTH_TRAVEL_PX:
        return False
    jitter = float(np.median(np.abs(np.diff(centers_px, n=2, axis=0)), axis=0).max())
    return travel / max(jitter, 1e-3) >= MIN_TRAVEL_TO_JITTER


def locate_precisely(client: CostGuardClient, scene: Scene, frames_jpeg: list[bytes], info: media.VideoInfo, run_id: str,
                     audience: str, record: Callable[..., None], limits: dict[str, Any],
                     progress: Progress) -> tuple[Scene, int]:
    """Boxes from one frame alone, and the frame SAM should start from (the first keyframe showing the object).

    The four-frame scene call names the object well but can place its box off by an object's width (on a real
    pendulum clip it sat on the stopwatch beside the bob); asked about a single frame, the same model boxes it tightly.
    A thrown object may not be in view yet at frame 0, so later keyframes are tried. Any failure keeps the scene's boxes.
    """
    cfg = limits.get("scene") or {}
    if not cfg.get("locate_single_image", False):
        return scene, 0
    progress(f"Locating the {scene.moving_object} precisely")
    obj = scene.moving_object
    reference = None if scene.reference_kind == "none" else scene.reference_kind.replace("_", " ")
    indices = media.keyframe_indices(info.frames, len(frames_jpeg))
    for frame_no, jpeg in list(zip(indices, frames_jpeg, strict=False))[: int(cfg.get("locate_max_frames", 3))]:
        try:
            loc, call = locate_objects(client, jpeg, obj, scene.second_object, reference, run_id, audience)
        except (LLMError, CostGuardError) as e:
            log.warning("locate skipped: %s", e)
            return scene, 0
        when = f"{frame_no / info.fps:.1f} s"
        if loc.moving_object_box is None or (scene.second_object and loc.second_object_box is None):
            record("locate", call, f"The {obj} is not in view at {when}; looks at a later frame.", status="rejected")
            continue
        update: dict[str, Any] = {"moving_object_box": loc.moving_object_box}
        if scene.second_object:
            update["second_object_box"] = loc.second_object_box
        if scene.reference_box is not None and loc.reference_box is not None:
            update["reference_box"] = loc.reference_box  # a reference that isn't found keeps its scene box: it doesn't move
        record("locate", call, f"Boxes the {obj} on the frame at {when}, where SAM starts tracking.", status="accepted")
        return scene.model_copy(update=update), frame_no
    return scene, 0


def track_objects(client: CostGuardClient, clip: Path, info: media.VideoInfo, scene: Scene, run_id: str,
                  audience: str, prompt_frame: int = 0) -> dict[str, ObjectTrack]:
    """Moving object (id 1) and reference (id 2) in ONE box-prompt request: half the SAM cost of two concept prompts."""
    f0 = prompt_frame
    objects = [msp.PromptObject(id=int(MOVING_ID), prompts=[msp.FramePrompt(frame=f0, box=_box(scene.moving_object_box, info))])]
    if scene.reference_box is not None:
        objects.append(msp.PromptObject(id=int(REFERENCE_ID), prompts=[msp.FramePrompt(frame=f0, box=_box(scene.reference_box, info))]))
    if scene.second_object_box is not None:
        objects.append(msp.PromptObject(id=int(SECOND_ID), prompts=[msp.FramePrompt(frame=f0, box=_box(scene.second_object_box, info))]))
    prompt = msp.build_object_prompt(size=(info.width, info.height), objects=objects)
    r = client.sam_video(prompt=prompt, video_path=clip, video_hash=file_digest(clip), est_frames=info.frames,
                         metadata=SAM_METADATA, run_id=run_id, audience=audience)
    try:
        return tracks_from_text(r.text, completed=r.finish_reason in (None, "completed"))
    except TrackingError as e:
        raise AnalysisError(str(e)) from e


NAMED_SAM_METADATA = {"mask_encoding": "one_bit", "include_confidence": "true"}  # concept prompts take no direction


def check_and_repair_track(client: CostGuardClient, clip: Path, info: media.VideoInfo, scene: Scene, moving: ObjectTrack,
                           run_id: str, audience: str, record: Callable[..., None], trail: list[dict[str, Any]],
                           progress: Progress) -> ObjectTrack:
    """Before any physics, the reasoning model looks at what SAM followed; if it's the wrong thing, SAM is asked again
    by name. A box placed a little off (on a clasp, a shadow, a digit drawn on the video) is the usual cause."""
    progress("Checking what was tracked")
    q = assess_track(moving, info.frames)
    numbers = {"tracked_frames": f"{q.coverage:.0%}", "largest_jump_vs_typical_step": round(q.jump_ratio, 1),
               "outline_size_spread": f"{q.size_spread:.0%}", "problems": q.problems}
    try:
        check, call = check_track(client, overlay_frames(str(clip), moving), scene.moving_object, numbers, run_id, audience)
    except (LLMError, CostGuardError) as e:  # a safeguard: without it the first track stands
        log.warning("Track check skipped: %s", e)
        return moving
    obj = scene.moving_object
    record("track_check", call, f"Looks at what SAM followed: {check.tracked_thing}. "
           + ("That is the object to measure." if check.on_target else f"That is not the {obj}, so it asks SAM again by name."),
           status="accepted" if check.on_target else "rejected", reasoning_summary=check.reasoning_summary)
    if check.on_target:
        return moving
    progress(f"Re-tracking the {obj} by name with SAM 3.1")
    named = track_by_name(client, clip, info, scene, run_id, audience)
    if named is None:
        trail.append({"step": "tracking", "provider": "meta_api", "model": "sam-3.1",
                      "text": f"SAM found nothing called {obj!r}; the first track is kept."})
        return moving
    trail.append({"step": "tracking", "provider": "meta_api", "model": "sam-3.1",
                  "text": f"Re-tracked the {obj} by name: {len(named.frames)} of {info.frames} frames."})
    return named


def track_by_name(client: CostGuardClient, clip: Path, info: media.VideoInfo, scene: Scene, run_id: str,
                  audience: str) -> ObjectTrack | None:
    """A concept prompt: SAM finds the object from its name, independent of any box. The match that moves most wins."""
    r = client.sam_video(prompt=scene.moving_object, video_path=clip, video_hash=file_digest(clip), est_frames=info.frames,
                         metadata=NAMED_SAM_METADATA, run_id=run_id, audience=audience)
    try:
        found = tracks_from_text(r.text, completed=r.finish_reason in (None, "completed"))
    except TrackingError:
        return None
    return max(found.values(), key=lambda t: (len(t.frames) >= 0.5 * info.frames, float(np.ptp(t.centers_px, axis=0).max())))


def _box(box, info: media.VideoInfo) -> msp.PromptBox:
    x1, y1, x2, y2 = box.to_pixels(info.width, info.height)
    return msp.PromptBox(x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))


def calibrate(scene: Scene, reference: ObjectTrack | None, moving: ObjectTrack, options: RunOptions) -> Calibration:
    if options.known_px_per_unit is not None:
        return Calibration(options.known_px_per_unit, 0.0, "known_scale")
    if reference is not None and scene.reference_kind != "none":
        try:
            return from_reference(reference, scene.reference_kind)
        except CalibrationError as e:
            if options.pendulum_length_m is None and options.object_size_m is None:
                raise AnalysisError(str(e)) from e
            log.warning("Reference calibration failed (%s); using the size given by the user.", e)
    if options.pendulum_length_m is not None and scene.motion_family == "pendulum":
        radius_px = fit_circle(moving.centers_px * np.array([1.0, -1.0]))[2]
        return from_known_length(radius_px, options.pendulum_length_m)
    if options.pendulum_length_m is not None:
        # A circle fitted to a spring's straight stroke would give a nonsense scale.
        log.warning("Ignoring the pendulum length: the clip shows %s motion.", scene.motion_family)
    if options.object_size_m is not None:
        # The object's own mask, measured in every tracked frame: its spread is reported as the scale's uncertainty.
        long_px = np.asarray(moving.long_px, dtype=float)
        size_px = float(np.median(long_px))
        return Calibration(size_px / options.object_size_m, float(np.std(long_px) / size_px), "object_size")
    # No scale: carry on. Frequency, damping and friction don't depend on scale; everything measured in
    # meters is reported as uncalibrated instead of the whole analysis being refused.
    log.info("No scale reference; continuing with a nominal scale (lengths and g will be marked uncalibrated).")
    return Calibration(NOMINAL_PX_PER_M, 0.0, "none")


def build_trajectory(moving: ObjectTrack, fps: float, px_per_m: float) -> tuple[Trajectory, np.ndarray]:
    """The collision-free stretch of the track with the most motion."""
    traj, origin = full_track(moving, fps, px_per_m)
    return longest_flight(traj, origin, px_per_m)


def full_track(moving: ObjectTrack, fps: float, px_per_m: float,
               origin_px: tuple[float, float] | None = None) -> tuple[Trajectory, np.ndarray]:
    segments = fill_short_gaps(moving.frames, moving.centers_px, max_gap=3)
    frames, centers = max(segments, key=lambda s: len(s[0]))
    if origin_px is not None:
        # Coordinates from a fixed point of the frame (the Sun in orbit mode) instead of the first sample.
        origin = np.asarray(origin_px, float)
        return from_pixels(frames, centers, fps, px_per_m, origin=origin), origin
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


def _hypothesis_items(hyp, sampled: SampledHypotheses | None = None) -> list[dict[str, Any]]:
    def votes(key: str) -> str:
        if sampled is None or sampled.samples <= 1:
            return ""
        return f" Proposed by {sampled.votes.get(key, 0)} of {sampled.samples} independent samples."
    items = [{"model_id": h.model_id, "title": MODELS[h.model_id].title, "why": h.rationale + votes(h.model_id)}
             for h in hyp.candidates]
    for f in hyp.free_models:
        law = "; ".join(f.acceleration)
        items.append({"model_id": f.title, "title": f.title,
                      "why": f"{f.rationale} Written by the agent: acceleration = {law}.{votes(f.title)}"})
    return items


TOOL_ACTIONS = ("test_models", "inspect")
FINAL_FOR = {"accepted": "explain", "ambiguous": "request_experiment", "no_fit": "give_up"}


def inspect_window(traj: Trajectory, turn: AgentTurn, uncalibrated: bool) -> dict[str, Any]:
    """The agent's 'inspect' tool: the same trajectory numbers it already reads, for a stretch of the clip."""
    w = turn.window
    if w is None:
        return {"text": "No window was given, so there was nothing to inspect.", "entry": {"error": "no window given"}}
    span = window_indices(traj.t, w.start_s, w.end_s, min_fraction=0.0, min_samples=10)
    if isinstance(span, str):
        return {"text": f"The window {w.start_s:.2f} to {w.end_s:.2f} s could not be used: {span}.",
                "entry": {"window": [w.start_s, w.end_s], "error": span}}
    numbers = summarize(traj.slice(*span), uncalibrated)
    return {"text": f"Inspects {w.start_s:.2f} to {w.end_s:.2f} s ({w.reason}).",
            "entry": {"window": [w.start_s, w.end_s], "numbers": numbers}}


def tool_outcome(before: Verdict, after: Verdict) -> str:
    if after is before:
        return "Nothing it named could be fitted to this motion, so the verdict is unchanged."
    if after.status == before.status and (after.winner is None) == (before.winner is None) and \
            (after.winner is None or after.winner.model_id == before.winner.model_id):
        return f"Fitted and tested: the verdict is unchanged ({after.status})."
    now = f"\"{MODELS[after.winner.model_id].title}\" accepted" if after.winner else after.status
    return f"Fitted and tested: the verdict changes from {before.status} to {now}."


def _winner_votes(s: SampledHypotheses, v: Verdict) -> dict[str, Any] | None:
    """How many independent proposals named the law that verification accepted (None without sampling or a winner)."""
    if s.samples <= 1 or v.winner is None:
        return None
    wid = v.winner.model_id
    n = s.votes.get(wid, s.votes.get(MODELS[wid].title, 0))
    return {"samples": s.samples, "winner": wid, "proposed_by": n, "families": s.families}


def _sampled_text(s: SampledHypotheses) -> str:
    h = s.merged
    n = len(h.candidates) + len(h.free_models)
    if s.samples <= 1:
        return f"Proposes {n} {h.family} models to test."
    agree = s.families.count(h.family)
    return (f"{s.samples} independent proposals ({agree} of {s.samples} chose {h.family}); "
            f"together they name {n} {h.family} models, all of which are tested.")


def _candidates(hyp) -> tuple[list[str], dict[str, str]]:
    """Listed models plus the agent's own free-form laws (already validated by the schema), with controls."""
    ids = [h.model_id for h in hyp.candidates]
    rationales = {h.model_id: h.rationale for h in hyp.candidates}
    for f in hyp.free_models:
        spec = register(build(hyp.family, f.title, f.acceleration, [FreeParam(**p.model_dump()) for p in f.params]))
        ids.append(spec.id)
        rationales[spec.id] = f"{f.rationale} (written by the agent)"
    return _with_control(hyp.family, ids), rationales


def _with_control(family: str, ids: list[str]) -> list[str]:
    """Always include the family's simplest model, so a hypothesis list can't hide the baseline.

    For motion outside the named families both controls are kept: a law the agent wrote must beat
    "nothing acts on it" and "a constant force" to be believed.
    """
    controls = [m.id for m in models_for(family) if m.id in GENERIC_CONTROLS] or \
        [min(models_for(family), key=lambda m: len(m.params)).id]
    return list(dict.fromkeys([*controls, *ids]))


GENERIC_CONTROLS = ("line_constant_velocity", "line_constant_accel", "plane_constant_velocity", "plane_constant_accel")


def agent_context(v: Verdict, family: str, summary: dict[str, Any], scene: Scene, tried: list[str]) -> dict[str, Any]:
    return {
        "verdict": v.status, "verdict_reason": v.reason, "family": family, "families_tried": tried,
        "object": scene.moving_object,
        "candidates": [fit_summary(f) for f in v.ranked],
        "remaining_contenders": [f.model_id for f in v.contenders],
        "trajectory_summary": summary,
    }


def fit_summary(f: FitResult, rationale: str | None = None) -> dict[str, Any]:
    spec = MODELS[f.model_id]
    return {"model_id": f.model_id, "title": spec.title, "params": f.params, "bic": f.bic,
            "heldout_nrmse": f.heldout_nrmse, "rmse": f.rmse, "rationale": rationale,
            "param_meta": [{"name": p.name, "latex": p.latex, "unit": p.unit} for p in spec.params]}


EARTH_G = 9.81
RULER_MAX_REL_STD = 0.1  # g must itself be well measured before it can serve as the ruler


def gravity_ruler_applies(winner: FitResult | None, family: str) -> bool:
    """No reference in view, but the verified law contains g: assume Earth and let gravity set the scale."""
    return (winner is not None and family in ("projectile", "pendulum") and winner.params.get("g", 0.0) > 0)


def length_power(unit: str) -> int:
    """How a quantity scales with length: m and m/s^2 once, 1/m inversely, angles and times not at all."""
    num, _, den = (unit or "").partition("/")
    return int(needs_scale(num)) - int(needs_scale(den))


def apply_gravity_ruler(result: dict[str, Any]) -> Calibration:
    """Rescale every length so the fitted g is 9.81 m/s^2; g becomes the assumption, its spread the scale's error.

    Pixels don't change, so ghosts and tracks drawn on the clip stay exactly where they were.
    """
    params = {p["name"]: p for p in result["parameters"]}
    g = params["g"]
    rel = g["std"] / g["value"]
    f = EARTH_G / g["value"]
    measured = round(g["value"], 3)

    def scaled(v: float, unit: str) -> float:
        return v * f ** length_power(unit)

    for p in result["parameters"]:
        k = length_power(p["unit"])
        if k:
            p["value"], p["std"] = scaled(p["value"], p["unit"]), abs(scaled(p["std"], p["unit"]))
            p["ci95"] = sorted(scaled(c, p["unit"]) for c in p["ci95"])
    g.update(value=EARTH_G, std=0.0, ci95=[EARTH_G, EARTH_G], derived="assumed: Earth's gravity, used as the ruler")
    for fit in result.get("ranking", []):
        units = {m["name"]: m["unit"] for m in fit.get("param_meta", [])}
        fit["params"] = {k: scaled(v, units.get(k, "")) for k, v in fit["params"].items()}
    fs = result.get("fit_state")
    if fs:
        units = {p.name: p.unit for p in spec_for(fs).params}
        fs["params"] = {k: scaled(v, units.get(k, "")) for k, v in fs["params"].items()}
        fs["ctx"] = {k: v * f if k in ("pivot_x", "pivot_y", "length_m", "length_std", "origin_x", "origin_y") else v
                     for k, v in fs["ctx"].items()}
        if result.get("family") == "projectile":
            fs["y0"] = [v * f for v in fs["y0"]]  # positions and velocities; a pendulum's angle needs no scale
        fs["px_per_m"] = fs["px_per_m"] / f
    fc = result.get("forecast")
    if fc and fc.get("unit") == "m":
        fc["measured"] = [v * f for v in fc["measured"]]
        fc["forecast"] = [v * f for v in fc["forecast"]]
    if result.get("bounces"):
        result["bounces"]["g"] *= f
        result["bounces"]["g_std"] *= f
    cal = Calibration(NOMINAL_PX_PER_M / f, rel, "gravity")
    result["calibration"] = {"px_per_m": cal.px_per_m, "rel_std": cal.rel_std, "source": cal.source}
    result["scale_basis"] = (f"No reference of known size was in view, so Earth's gravity is the ruler: the fit gave "
                             f"g = {measured} in the clip's own units, and every length is scaled so that g = 9.81 m/s² "
                             f"(scale uncertainty {rel:.1%}). Filmed somewhere other than Earth, lengths would be off.")
    result["trail"].append({
        "step": "plausibility", "provider": None, "model": "simulate-and-compare", "status": "accepted",
        "text": f"With no reference in view, g came out as {measured} in arbitrary units. Assuming the clip was filmed on "
                f"Earth, gravity becomes the ruler: lengths are scaled by {f:.3g} so that g = 9.81 m/s², with a "
                f"{rel:.1%} scale uncertainty carried into every length."})
    return cal


def add_scale_error(parameters: list[dict[str, Any]], rel_std: float) -> None:
    """The scale's own spread (measured frame to frame on the reference) is part of every length's error."""
    if rel_std <= 0:
        return
    for p in parameters:
        if needs_scale(p["unit"]) and not p.get("derived"):
            extra = abs(p["value"]) * rel_std
            lo, hi = p["ci95"]
            p["std"] = float(np.hypot(p["std"], extra))
            p["ci95"] = [p["value"] - float(np.hypot(p["value"] - lo, 1.96 * extra)),
                         p["value"] + float(np.hypot(hi - p["value"], 1.96 * extra))]
            p["includes_scale_error"] = True


def known_length_geometry(moving: ObjectTrack, origin_px: np.ndarray, calibration: Calibration,
                          options: RunOptions) -> dict[str, float] | None:
    """With the length given, the swing is measured on that length rather than on a re-fitted arc.

    With the pivot out of frame a short arc's radius and centre are poorly pinned down (on a real clip two
    stretches of one swing gave radii 50% apart), and g would move with them.
    """
    if calibration.source != "user" or options.pendulum_length_m is None:
        return None
    ppm = calibration.px_per_m
    x = (moving.centers_px[:, 0] - origin_px[0]) / ppm
    y = (origin_px[1] - moving.centers_px[:, 1]) / ppm
    # The pivot hangs straight above the rest point (the middle of the swing, at the bottom of the arc), so the
    # angle swings about zero; with the length known, g then rests on the period alone.
    rest_x = float(np.percentile(x, 5) + np.percentile(x, 95)) / 2
    bottom_y = float(np.percentile(y, 2))
    L = options.pendulum_length_m
    return {"pivot_x": rest_x, "pivot_y": bottom_y + L, "length_m": L, "length_std": 0.0, "length_given": 1.0}


def derived_parameters(c: Coordinates) -> list[dict[str, Any]]:
    if c.family == "pendulum":
        L, s = c.ctx["length_m"], c.ctx["length_std"]
        how = "given by you; g then follows from the period" if c.ctx.get("length_given") else "radius of the fitted swing arc"
        return [{"name": "L", "latex": "L", "unit": "m", "value": L, "std": s, "ci95": [L - 1.96 * s, L + 1.96 * s],
                 "derived": how}]
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
