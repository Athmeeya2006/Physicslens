"""The three things the reasoning model does: read the scene, propose hypotheses, decide the next step."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..costguard import CostGuardClient
from ..physics.models import MODELS
from .llm import LLMCall, ask_json
from .schemas import (
    AgentTurn,
    Hypotheses,
    Located,
    Scene,
    SkepticReview,
    StudyPlan,
    StudyTurn,
    TrackCheck,
)

SCENE_SYSTEM = """You look at up to four frames from a short phone video of something moving, for a physics measurement tool.
Identify the single object whose motion should be measured and a reference object of known size in the same plane.
When a person pushes, throws, drops or releases something, measure that thing (the truck, the ball, the door), not the person.
- moving_object: a short concrete noun phrase (1-4 words) naming only the object, e.g. "metal ball", "red weight". It is sent to a segmentation model, so no commands, lists, colors of other things, or positions.
- Boxes are on a 0-1000 grid relative to the FIRST image (x right, y down) and must tightly enclose the object in that first image.
- motion_family: pendulum (swings on a string or rod), oscillator (bounces on a spring or elastic, back and forth along a line), projectile (thrown, dropped or falling), double_pendulum (a second arm hinged to the end of the first), rotation (spins in place about its own centre and slows down, e.g. a spun bottle, wheel or bar), other (anything else that moves: rolling, sliding, braking, a toy car, a falling leaf).
- For a double_pendulum, moving_object/moving_object_box is the INNER bob (nearest the pivot) and second_object/second_object_box the OUTER bob; otherwise both are null.
- reference_kind: only an object from the allowed list that is clearly visible and roughly flat facing the camera; otherwise "none". Never guess a size."""

HYPOTHESES_SYSTEM = """You propose candidate equations of motion for a measured trajectory. You do not decide which is correct:
every candidate you name will be fitted and simulated, and only numerical verification on held-out data decides.
Pick the family that matches the numbers, then 1 to 4 candidate model ids from that family that are physically plausible.
Prefer including the simplest model plus the variants the evidence suggests (e.g. damping if amplitude decays).
Available models:
""" + "\n".join(f"- {m.id} ({m.family}): {m.title}" for m in MODELS.values() if not m.id.startswith("custom_")) + """

Motion that none of the named families describes (a cart braking, a ball rolling down a ramp, a falling leaf, a
sliding block, a puck slowing on ice) uses family "line" (moving back and forth or onward along one straight path)
or "plane" (curving freely in the image plane). For these, pick line_/plane_ controls as candidates and write up to
three laws of your own in free_models. Each law gives the acceleration in a closed math language:
- line: one expression in s (position along the path, m; it increases to the right, or upward when the path is
  closer to vertical) and v (its velocity, m/s).
- plane: two expressions, x then y acceleration, in x, y (m, y up) and vx, vy (m/s).
- Operators + - * / ^ and parentheses; functions sin cos tan tanh atan exp log sqrt abs sign hypot; constant pi.
  Every other name must be one of your declared params (at most 4), each with an SI unit, bounds and a guess;
  every param is fitted to the data. Write a number instead only where two params could not be told apart.
- Example, a cart slowed by rolling friction and air drag: ["-mu*9.81*sign(v) - c*v*abs(v)"] with params mu (unit
  empty) and c (1/m); mu and g only ever appear as a product, so g is written as 9.81.
Write the physics you actually believe, simplest first. Your laws are fitted and tested on held-out motion
against the controls exactly like the listed models; one that does not beat them is rejected.
For the named families, free_models must be [].
If the start or end of the clip is clearly not free motion (a hand still throwing, holding or catching the object),
give fit_window with the free stretch in seconds and why; otherwise null. Code checks it: a window that drops more
than 30% of the clip is refused, and the whole clip is fitted too, so cutting can't hide a bad fit."""

AGENT_SYSTEM = """You are the experiment-planning agent of a system-identification tool. Numerical verification has already
fitted and simulated candidate equations against the measured motion; its verdict is final and you must not contradict it.
Allowed actions depend on the verdict:
- accepted: action "explain". Explain in at most three plain sentences what the equation and fitted parameters mean physically for this object, for an engineer.
- ambiguous: action "request_experiment". Ask the user for ONE concrete new recording that would separate the remaining candidates, and say why in one sentence. If "experiment_design" is given, it was computed by simulating every tied equation under candidate experiments: request its recommended experiment and quote its predicted separation versus the tracking noise. Do not invent a different experiment.
- no_fit: action "try_family" with next_family if the numbers clearly suggest a different family that has not been tried yet; otherwise "give_up" and explain briefly that no clean governing equation was found and what might help.
Before that final action you may run tool steps while "tool_steps_left" is above zero; each result appears in
"notebook" on your next turn:
- test_models: up to 3 listed model ids of the current family from "untested_models" (or, for family line or
  plane, one law of your own in free_model) that the numbers suggest could forecast better. Code fits them next to
  the others and recomputes the verdict by the same rule; you cannot change it any other way.
- inspect: a window (start_s, end_s, reason) of the clip; you get its trajectory numbers (e.g. to see whether an
  amplitude decays exponentially or linearly, or whether the start is still a hand throwing).
For a tool step, give "expectation": what you expect and what result would change your mind. Use tools when the
evidence points at something untested; when the verdict is already clear, go straight to the final action.
Never claim an equation is correct beyond what the verdict says. Never invent numbers.
If "scale" says none, the clip had no reference of known size: do not quote lengths, amplitudes or g as measured
values; talk about frequency, damping and friction, which are exact."""


def understand_scene(client: CostGuardClient, keyframes: list[bytes], run_id: str, audience: str = "dev",
                     run_cap_usd: float | None = None, hint: str | None = None) -> tuple[Scene, LLMCall]:
    text = (f"{len(keyframes)} frames from the clip, first frame first. "
            "Return the JSON for the moving object and reference object.")
    if hint:
        # The user's own words about what to measure: a busy scene can have several candidates.
        text += f" The user says the object to measure is: {json.dumps(hint)}."
    return ask_json(client, task="scene", system=SCENE_SYSTEM, text=text, schema_model=Scene, run_id=run_id,
                    images=keyframes, audience=audience, run_cap_usd=run_cap_usd)


def propose_hypotheses(client: CostGuardClient, summary: dict[str, Any], scene: Scene, run_id: str,
                       audience: str = "dev", exclude_families: list[str] | None = None,
                       run_cap_usd: float | None = None, sample: tuple[int, int] | None = None) -> tuple[Hypotheses, LLMCall]:
    payload = {"scene": {"object": scene.moving_object, "family_guess_from_video": scene.motion_family,
                         "observations": scene.observations},
               "trajectory_summary": summary, "families_already_ruled_out": exclude_families or []}
    if sample is not None:
        payload["independent_proposal"] = f"{sample[0]} of {sample[1]}: reason it through on your own"
    return ask_json(client, task="hypotheses", system=HYPOTHESES_SYSTEM, text=json.dumps(payload), schema_model=Hypotheses,
                    run_id=run_id, audience=audience, run_cap_usd=run_cap_usd)


@dataclass(frozen=True)
class SampledHypotheses:
    merged: Hypotheses  # the majority family's samples, candidates and laws in first-seen order
    calls: list[LLMCall]
    votes: dict[str, int]  # model id or law title -> how many samples proposed it
    samples: int
    families: list[str]


def propose_hypotheses_sampled(client: CostGuardClient, summary: dict[str, Any], scene: Scene, run_id: str,
                               audience: str = "dev", exclude_families: list[str] | None = None,
                               samples: int = 1) -> SampledHypotheses:
    """Several independent proposals in parallel; the union of the majority family's candidates is fitted.

    Self-consistency costs little here: fitting is local, and how many samples proposed the law that wins is a
    direct, reportable measure of how sure the model was before the math looked.
    """
    if samples <= 1:
        hyp, call = propose_hypotheses(client, summary, scene, run_id, audience, exclude_families)
        return SampledHypotheses(hyp, [call], _votes([hyp]), 1, [hyp.family])
    with ThreadPoolExecutor(samples) as pool:
        # A distinct note per sample keeps them independent: identical prompts would just replay one cached answer.
        futures = [pool.submit(propose_hypotheses, client, summary, scene, run_id, audience, exclude_families,
                               sample=(i + 1, samples)) for i in range(samples)]
        results = [f.result() for f in futures]
    hyps = [h for h, _ in results]
    families = [h.family for h in hyps]
    family = max(families, key=lambda f: (families.count(f), -families.index(f)))
    chosen = [h for h in hyps if h.family == family]
    candidates = list({c.model_id: c for h in chosen for c in h.candidates}.values())
    laws = list({f.title: f for h in chosen for f in h.free_models}.values())[:MAX_FREE_LAWS]
    window = next((h.fit_window for h in chosen if h.fit_window is not None), None)
    # Built without re-validating the list lengths: the union may hold more than one sample is allowed to propose.
    merged = Hypotheses.model_construct(family=family, candidates=candidates, free_models=laws, fit_window=window)
    return SampledHypotheses(merged, [c for _, c in results], _votes(chosen), samples, families)


MAX_FREE_LAWS = 4


def _votes(hyps: list[Hypotheses]) -> dict[str, int]:
    votes: dict[str, int] = {}
    for h in hyps:
        for key in {c.model_id for c in h.candidates} | {f.title for f in h.free_models}:
            votes[key] = votes.get(key, 0) + 1
    return votes


def agent_turn(client: CostGuardClient, context: dict[str, Any], run_id: str, audience: str = "dev") -> tuple[AgentTurn, LLMCall]:
    return ask_json(client, task="agent", system=AGENT_SYSTEM, text=json.dumps(context), schema_model=AgentTurn,
                    run_id=run_id, audience=audience, meta_effort="minimal")


TRACK_CHECK_SYSTEM = """You check a measurement before any physics is fitted to it. The image shows three frames of a
clip (start, middle, end) side by side. In each, a green box marks what the segmentation model tracked as the moving
object. Say whether the green box surrounds the named object itself in all three frames. A box on a part of it (a hook,
a clasp, a shadow), on something else, or on text drawn over the video is off target. Numbers about the track's
smoothness are given as extra evidence. Do not judge the physics."""


def check_track(client: CostGuardClient, overlay_jpeg: bytes, object_name: str, quality: dict[str, Any], run_id: str,
                audience: str = "dev") -> tuple[TrackCheck, LLMCall]:
    payload = {"object": object_name, "track_numbers": quality}
    return ask_json(client, task="track_check", system=TRACK_CHECK_SYSTEM, text=json.dumps(payload),
                    schema_model=TrackCheck, run_id=run_id, images=[overlay_jpeg], audience=audience,
                    meta_effort="minimal", max_output_tokens=800)


STUDY_PLAN_SYSTEM = """You plan a measurement study for an engineer who films things with a phone. The tool can identify:
swinging (pendulums, hanging loads, doors on hinges), springs and bouncing suspensions, thrown or falling objects
with drag, bouncing balls, spin-down of wheels, fans and bearings, double pendulums, and anything sliding or rolling
to a stop (the agent can write its own law for that). Each recording is analyzed separately and verified on
held-out motion; you then combine them. Given the user's goal, choose the deliverable, the quantities to measure,
and 1 to 4 recordings with concrete filming instructions (camera still and side-on, the whole motion in view, an
A4 sheet or an object of known size in the plane of motion when lengths matter, slow motion for fast things).
Ask for repeat recordings when the goal needs a trustworthy number, and for recordings spread over time for wear.
When the goal is how a quantity depends on a setting the user can change (period against string length, spring
frequency against added mass), plan a sweep: 3 to 4 settings spread over at least a factor of three, one recording
each (label it with the setting), the response to compare, and the law you expect. Code fits every candidate law
to the recordings and judges it by predicting each recording from the others; it then predicts the next setting
before it is filmed."""

STUDY_TURN_SYSTEM = """You run a measurement study toward the user's goal. You see the plan, every recording's verified
result, and the combined numbers (a weighted mean per quantity and whether the recordings agree within their error
bars; that test is final), and for a sweep the law across recordings that code accepted (or the tie, with the
setting that would separate it) and its prediction for the next setting. Decide: "finish" when the goal is answered, or "record_more" with exactly one concrete
recording that would answer what is still open (a failed recording, recordings that disagree, a quantity not yet
measured). Never invent numbers; quote only what is given. For a health_check, write a maintenance_note on what the
change (or no change) means for the machine and suggest next_check; otherwise leave both null."""


def plan_study(client: CostGuardClient, goal: str, run_id: str, audience: str = "dev") -> tuple[StudyPlan, LLMCall]:
    return ask_json(client, task="study_plan", system=STUDY_PLAN_SYSTEM, text=json.dumps({"goal": goal}),
                    schema_model=StudyPlan, run_id=run_id, audience=audience, meta_effort="low", max_output_tokens=2000)


def study_turn(client: CostGuardClient, context: dict[str, Any], run_id: str, audience: str = "dev") -> tuple[StudyTurn, LLMCall]:
    return ask_json(client, task="study_turn", system=STUDY_TURN_SYSTEM, text=json.dumps(context, default=str),
                    schema_model=StudyTurn, run_id=run_id, audience=audience, meta_effort="low", max_output_tokens=1500)


SKEPTIC_SYSTEM = """You are the referee. Numerical verification has accepted an equation of motion for a measured
trajectory. Your only job is to try to break it: propose up to three alternative laws that could explain the same
numbers as well or better. Each one is fitted and tested by code on motion it never saw; if one forecasts better by
the same rule (held-out error, then BIC), the verdict changes. You do not decide anything yourself.
- Use listed model ids from the SAME family as the accepted law that were not already tested.
- For family line or plane you may instead write a law (free_model) in the same closed math language as the
  hypotheses step: acceleration in s and v (line) or x, y, vx, vy (plane), at most 4 fitted params.
- Aim at real weaknesses: damping the accepted law leaves out, a nonlinearity at large amplitude, friction, drag.
- Put confounds code cannot test from one clip (camera tilt, a moving pivot, a scale reference out of the plane)
  in concerns, each tied to a number you were given. Never invent numbers. Do not repeat a model already tested."""


def skeptic_review(client: CostGuardClient, context: dict[str, Any], run_id: str,
                   audience: str = "dev") -> tuple[SkepticReview, LLMCall]:
    return ask_json(client, task="skeptic", system=SKEPTIC_SYSTEM, text=json.dumps(context, default=str),
                    schema_model=SkepticReview, run_id=run_id, audience=audience, max_output_tokens=2000)


LOCATE_SYSTEM = """You place boxes for a segmentation model on ONE video frame. Give a tight box around each named
object on a 0-1000 grid relative to this image (x right, y down), enclosing the whole object and nothing else
(not its string, hook, shadow, or anything next to it). If an object is not visible in this frame, give null."""


def locate_objects(client: CostGuardClient, frame_jpeg: bytes, moving: str, second: str | None, reference: str | None,
                   run_id: str, audience: str = "dev") -> tuple[Located, LLMCall]:
    """One frame alone: boxes from a single image are much tighter than from the four-frame scene call."""
    names = {"moving_object": moving, "second_object": second, "reference_object": reference}
    return ask_json(client, task="locate", system=LOCATE_SYSTEM, text=json.dumps({k: v for k, v in names.items() if v}),
                    schema_model=Located, run_id=run_id, images=[frame_jpeg], audience=audience, max_output_tokens=800)
