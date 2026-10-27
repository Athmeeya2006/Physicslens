"""The three things the reasoning model does: read the scene, propose hypotheses, decide the next step."""

from __future__ import annotations

import json
from typing import Any

from ..costguard import CostGuardClient
from ..physics.models import MODELS
from .llm import LLMCall, ask_json
from .schemas import AgentTurn, Hypotheses, Scene

SCENE_SYSTEM = """You look at up to four frames from a short phone video of something moving, for a physics measurement tool.
Identify the single object whose motion should be measured and a reference object of known size in the same plane.
- moving_object: a short concrete noun phrase (1-4 words) naming only the object, e.g. "metal ball", "red weight". It is sent to a segmentation model, so no commands, lists, colors of other things, or positions.
- Boxes are on a 0-1000 grid relative to the FIRST image (x right, y down) and must tightly enclose the object in that first image.
- motion_family: pendulum (swings on a string or rod), oscillator (bounces on a spring or elastic, back and forth along a line), projectile (thrown, dropped or falling), other.
- reference_kind: only an object from the allowed list that is clearly visible and roughly flat facing the camera; otherwise "none". Never guess a size."""

HYPOTHESES_SYSTEM = """You propose candidate equations of motion for a measured trajectory. You do not decide which is correct:
every candidate you name will be fitted and simulated, and only numerical verification on held-out data decides.
Pick the family that matches the numbers, then 1 to 4 candidate model ids from that family that are physically plausible.
Prefer including the simplest model plus the variants the evidence suggests (e.g. damping if amplitude decays).
Available models:
""" + "\n".join(f"- {m.id} ({m.family}): {m.title}" for m in MODELS.values())

AGENT_SYSTEM = """You are the experiment-planning agent of a system-identification tool. Numerical verification has already
fitted and simulated candidate equations against the measured motion; its verdict is final and you must not contradict it.
Allowed actions depend on the verdict:
- accepted: action "explain". Explain in at most three plain sentences what the equation and fitted parameters mean physically for this object, for an engineer.
- ambiguous: action "request_experiment". Ask the user for ONE concrete new recording that would separate the remaining candidates (e.g. a larger swing angle, a longer recording, a different drop height), and say why in one sentence.
- no_fit: action "try_family" with next_family if the numbers clearly suggest a different family that has not been tried yet; otherwise "give_up" and explain briefly that no clean governing equation was found and what might help.
Never claim an equation is correct beyond what the verdict says. Never invent numbers."""


def understand_scene(client: CostGuardClient, keyframes: list[bytes], run_id: str, audience: str = "dev") -> tuple[Scene, LLMCall]:
    text = (f"{len(keyframes)} frames from the clip, first frame first. "
            "Return the JSON for the moving object and reference object.")
    return ask_json(client, task="scene", system=SCENE_SYSTEM, text=text, schema_model=Scene, run_id=run_id,
                    images=keyframes, audience=audience)


def propose_hypotheses(client: CostGuardClient, summary: dict[str, Any], scene: Scene, run_id: str,
                       audience: str = "dev", exclude_families: list[str] | None = None) -> tuple[Hypotheses, LLMCall]:
    payload = {"scene": {"object": scene.moving_object, "family_guess_from_video": scene.motion_family,
                         "observations": scene.observations},
               "trajectory_summary": summary, "families_already_ruled_out": exclude_families or []}
    return ask_json(client, task="hypotheses", system=HYPOTHESES_SYSTEM, text=json.dumps(payload), schema_model=Hypotheses,
                    run_id=run_id, audience=audience)


def agent_turn(client: CostGuardClient, context: dict[str, Any], run_id: str, audience: str = "dev") -> tuple[AgentTurn, LLMCall]:
    return ask_json(client, task="agent", system=AGENT_SYSTEM, text=json.dumps(context), schema_model=AgentTurn,
                    run_id=run_id, audience=audience, meta_effort="minimal")
