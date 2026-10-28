"""The three things the reasoning model does: read the scene, propose hypotheses, decide the next step."""

from __future__ import annotations

import json
from typing import Any

from ..costguard import CostGuardClient
from ..physics.models import MODELS
from .llm import LLMCall, ask_json
from .schemas import AgentTurn, Hypotheses, Scene, TrackCheck

SCENE_SYSTEM = """You look at up to four frames from a short phone video of something moving, for a physics measurement tool.
Identify the single object whose motion should be measured and a reference object of known size in the same plane.
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
For the named families, free_models must be []."""

AGENT_SYSTEM = """You are the experiment-planning agent of a system-identification tool. Numerical verification has already
fitted and simulated candidate equations against the measured motion; its verdict is final and you must not contradict it.
Allowed actions depend on the verdict:
- accepted: action "explain". Explain in at most three plain sentences what the equation and fitted parameters mean physically for this object, for an engineer.
- ambiguous: action "request_experiment". Ask the user for ONE concrete new recording that would separate the remaining candidates, and say why in one sentence. If "experiment_design" is given, it was computed by simulating every tied equation under candidate experiments: request its recommended experiment and quote its predicted separation versus the tracking noise. Do not invent a different experiment.
- no_fit: action "try_family" with next_family if the numbers clearly suggest a different family that has not been tried yet; otherwise "give_up" and explain briefly that no clean governing equation was found and what might help.
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
                       run_cap_usd: float | None = None) -> tuple[Hypotheses, LLMCall]:
    payload = {"scene": {"object": scene.moving_object, "family_guess_from_video": scene.motion_family,
                         "observations": scene.observations},
               "trajectory_summary": summary, "families_already_ruled_out": exclude_families or []}
    return ask_json(client, task="hypotheses", system=HYPOTHESES_SYSTEM, text=json.dumps(payload), schema_model=Hypotheses,
                    run_id=run_id, audience=audience, run_cap_usd=run_cap_usd)


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
