"""Self-consistency: independent hypothesis samples, the majority family's union fitted, votes reported."""

import json

from physicslens.agent.reasoning import propose_hypotheses_sampled
from physicslens.agent.schemas import Scene
from physicslens.costguard.mock import responses_result

from .test_pipeline import SCENE


def hyp(family, *ids):
    return responses_result(text=json.dumps({"family": family, "free_models": [], "fit_window": None,
                                             "candidates": [{"model_id": m, "rationale": "r"} for m in ids]}))


def test_majority_family_union_and_votes(harness):
    harness.sdk().script.extend([hyp("pendulum", "pendulum", "pendulum_linear_damping"),
                                 hyp("pendulum", "pendulum", "pendulum_quadratic_damping"),
                                 hyp("oscillator", "harmonic")])
    s = propose_hypotheses_sampled(harness.client, {"n": 1}, Scene(**SCENE), "r", samples=3)
    assert s.samples == 3 and len(s.calls) == 3 and sorted(s.families) == ["oscillator", "pendulum", "pendulum"]
    assert s.merged.family == "pendulum"
    assert {c.model_id for c in s.merged.candidates} == {"pendulum", "pendulum_linear_damping", "pendulum_quadratic_damping"}
    assert s.votes["pendulum"] == 2 and s.votes["pendulum_linear_damping"] == 1 and "harmonic" not in s.votes
    # Each sample was asked separately (a shared prompt would just replay one cached answer).
    prompts = {c["input"][0]["content"][0]["text"] for c in harness.sdk().calls}
    assert len(prompts) == 3 and all("independent_proposal" in p for p in prompts)


def test_one_sample_is_the_plain_call(harness):
    harness.sdk().script.append(hyp("pendulum", "pendulum"))
    s = propose_hypotheses_sampled(harness.client, {"n": 1}, Scene(**SCENE), "r", samples=1)
    assert s.samples == 1 and s.votes == {"pendulum": 1}
    assert "independent_proposal" not in harness.sdk().calls[0]["input"][0]["content"][0]["text"]


def test_a_failed_sample_is_left_out_not_fatal(harness):
    """Real runs: one sample ran out of tokens, another wrote laws the grammar rejects; the others still count."""
    # Two good answers; whichever thread asks third gets the mock's default (not JSON) twice: order cannot matter.
    harness.sdk().script.extend([hyp("pendulum", "pendulum"), hyp("pendulum", "pendulum_linear_damping")])
    s = propose_hypotheses_sampled(harness.client, {"n": 1}, Scene(**SCENE), "r", samples=3)
    assert s.samples == 2 and s.failed == 1 and s.merged.family == "pendulum"


def test_all_samples_failing_still_fails_clearly(harness):
    import pytest
    from physicslens.agent.llm import LLMError
    harness.sdk().script.extend([responses_result(text="no")] * 6)
    with pytest.raises(LLMError):
        propose_hypotheses_sampled(harness.client, {"n": 1}, Scene(**SCENE), "r", samples=3)
