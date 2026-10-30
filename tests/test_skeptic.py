"""The referee proposes alternatives to an accepted law; only fitting and held-out verification can change the verdict."""

import json

import pytest
from physicslens.agent.schemas import Scene
from physicslens.agent.skeptic import challenge, outcome_text
from physicslens.costguard.mock import responses_result
from physicslens.physics.coords import to_coordinates
from physicslens.physics.fit import fit_model
from physicslens.physics.models import MODELS
from physicslens.physics.verify import assess

from .test_physics import pendulum_xy, track
from .test_pipeline import SCENE

FIRST = ("pendulum_small_angle", "pendulum")  # what the hypotheses step tested before the referee


def accepted_undamped(gamma: float):
    c = to_coordinates(track(pendulum_xy(0.9, gamma, 10)), "pendulum")
    v = assess(c, [fit_model(MODELS[m], c) for m in FIRST])
    assert v.status == "accepted" and v.winner.model_id == "pendulum"
    return c, v


def review(*challenges, concerns=()):
    return responses_result(text=json.dumps({"challenges": list(challenges), "concerns": list(concerns),
                                             "reasoning_summary": "Looked for damping the accepted law leaves out."}))


DAMPING = {"model_id": "pendulum_linear_damping", "free_model": None, "why": "The swing may decay slowly."}


def test_a_better_alternative_overturns_the_verdict(harness):
    c, v = accepted_undamped(0.02)
    harness.sdk().script.append(review(DAMPING))
    ch = challenge(harness.client, c, v, "pendulum", {"n": 1}, Scene(**SCENE), "r")
    assert not ch.info["survived"] and ch.info["tested"] == 1
    assert ch.verdict.status == "accepted" and ch.verdict.winner.model_id == "pendulum_linear_damping"
    assert ch.info["verdict_after"] == {"status": "accepted", "winner": "pendulum_linear_damping"}
    assert "verdict changes" in outcome_text(ch.info, "Simple pendulum")
    # The model saw the numbers it was asked to attack, and the untested models it could pick from.
    sent = json.loads(harness.sdk().calls[0]["input"][0]["content"][0]["text"])
    assert sent["accepted_law"]["model_id"] == "pendulum"
    assert "pendulum_linear_damping" in {m["model_id"] for m in sent["untested_models_in_family"]}


def test_the_law_survives_when_the_alternative_does_not_forecast_better(harness):
    c, v = accepted_undamped(0.0)
    harness.sdk().script.append(review(DAMPING, concerns=["The pivot may move slightly."]))
    ch = challenge(harness.client, c, v, "pendulum", {"n": 1}, Scene(**SCENE), "r")
    assert ch.info["survived"] and ch.verdict is not v and ch.verdict.winner.model_id == "pendulum"
    assert ch.info["concerns"] == ["The pivot may move slightly."]
    assert "survived 1 challenge" in outcome_text(ch.info, "Simple pendulum")
    item = ch.info["challenges"][0]
    assert item["model_id"] == "pendulum_linear_damping" and item["status"] != "accepted" and item["heldout_nrmse"] is not None


@pytest.mark.parametrize("proposal", [
    {"model_id": "damped_harmonic", "free_model": None, "why": "Maybe a spring."},                 # another family
    {"model_id": "pendulum", "free_model": None, "why": "Again."},                                  # already tested
    {"model_id": "made_up_model", "free_model": None, "why": "Invented."},                          # not a model
    {"model_id": None, "why": "Written law.", "free_model": {                                      # laws only for line/plane
        "title": "Odd law", "acceleration": ["-k*s"], "rationale": "r",
        "params": [{"name": "k", "unit": "1/s^2", "lower": 0, "upper": 100, "guess": 10}]}},
])
def test_untestable_proposals_never_touch_the_verdict(harness, proposal):
    c, v = accepted_undamped(0.02)
    harness.sdk().script.append(review(proposal))
    ch = challenge(harness.client, c, v, "pendulum", {"n": 1}, Scene(**SCENE), "r")
    assert ch.info["tested"] == 0 and ch.verdict is v and ch.info["survived"]
    assert ch.info["challenges"][0]["status"] == "untestable"
    assert "proposed nothing code could test" in outcome_text(ch.info, "Simple pendulum")


def test_pipeline_records_the_referee_in_the_trail_and_result(harness, tmp_path):
    from physicslens.pipeline import RunOptions, analyze

    from .test_pipeline import (
        AGENT,
        CLIP,
        HYPOTHESES,
        TRACK_OK,
        sam_result,
        sam_text_as_object_1,
        script,
    )
    script(harness, responses_result(text=json.dumps(SCENE)), sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(TRACK_OK)), responses_result(text=json.dumps(HYPOTHESES)),
           responses_result(text=json.dumps(AGENT)), review(DAMPING))
    result = analyze(harness.client, CLIP, tmp_path / "run", "run-ref", RunOptions(pendulum_length_m=0.6))
    if result["status"] != "accepted":
        pytest.skip("the referee runs only after an accepted law")
    steps = [t["step"] for t in result["trail"]]
    assert "skeptic" in steps and steps.index("skeptic") > steps.index("verification")
    ref = result["trail"][steps.index("skeptic")]
    assert ref["provider"] == "meta_api" and ref["items"] and ref["status"] in ("accepted", "rejected")
    assert result["skeptic"]["accepted_before"] and "survived" in result["skeptic"]
    assert "skeptic" in [c["step"] for c in result["llm_calls"]]


def test_a_batch_can_turn_the_referee_off_and_the_trail_says_so(harness, tmp_path):
    from physicslens.pipeline import RunOptions, analyze

    from .test_pipeline import AGENT, CLIP, HYPOTHESES, TRACK_OK, sam_result, sam_text_as_object_1, script
    script(harness, responses_result(text=json.dumps(SCENE)), sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(TRACK_OK)), responses_result(text=json.dumps(HYPOTHESES)),
           responses_result(text=json.dumps(AGENT)))
    harness.client.limits["agent"] = {**(harness.client.limits.get("agent") or {}), "referee": False}
    result = analyze(harness.client, CLIP, tmp_path / "run", "run-noref", RunOptions(pendulum_length_m=0.6))
    if result["status"] != "accepted":
        pytest.skip("the referee runs only after an accepted law")
    ref = [t for t in result["trail"] if t["step"] == "skeptic"]
    assert len(ref) == 1 and ref[0]["status"] == "skipped" and "turned it off" in ref[0]["text"]
    assert result.get("skeptic") is None and "skeptic" not in [c["step"] for c in result["llm_calls"]]
