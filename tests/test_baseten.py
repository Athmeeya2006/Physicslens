"""Muse Spark Open on Baseten: the primary open-weight reasoning path, its fallback order and its labels."""

import json
from types import SimpleNamespace

import pytest
import yaml
from physicslens.agent.llm import OpenWeightUnavailable, ask_json
from physicslens.agent.schemas import Scene
from physicslens.costguard import PriceMissing
from physicslens.costguard.client import _parse_chat, split_reasoning
from physicslens.costguard.mock import api_error, chat_result
from physicslens.settings import BASETEN_MODEL, Settings

from .conftest import Harness
from .test_pipeline import SCENE


@pytest.fixture
def baseten_env(monkeypatch):
    monkeypatch.setenv("REASONING_PROVIDER", "baseten")
    monkeypatch.setenv("BASETEN_API_KEY", "test-not-a-real-key")
    monkeypatch.delenv("GLIMMER_BASE_URL", raising=False)


def env_harness(settings, app_env: str) -> Harness:
    return Harness(Settings(app_env=app_env, repo_root=settings.repo_root, config_dir=settings.config_dir,
                            var_dir=settings.var_dir))


def test_baseten_used_when_selected(settings, baseten_env):
    h = Harness(settings)
    h.sdk("baseten").script.append(chat_result(text=json.dumps(SCENE)))
    scene, call = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert call.provider == "baseten" and call.model == BASETEN_MODEL == "meta-models/muse-spark-open-1026"
    assert call.reasoning == "mock reasoning" and call.fallbacks == ()
    assert call.cost_usd > 0  # paid provider: priced and recorded in the ledger
    assert scene.moving_object == "red ball"
    sent = h.sdk("baseten").calls[0]
    # Muse Spark Open defaults to xhigh effort; the effort for each step is always sent, and sampling is left alone.
    assert sent["extra_body"] == {"reasoning_effort": "low"}
    assert sent["model"] == BASETEN_MODEL and "temperature" not in sent
    assert h.sdk("meta_api").calls == [] and h.sdk("glimmer").calls == []


@pytest.mark.parametrize("app_env", ["demo", "prod"])
def test_demo_and_prod_reason_on_baseten(settings, baseten_env, app_env):
    h = env_harness(settings, app_env)
    h.sdk("baseten").script.append(chat_result(text=json.dumps(SCENE)))
    _, call = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert call.provider == "baseten"


@pytest.mark.parametrize("app_env", ["demo", "prod"])
def test_baseten_failure_never_reaches_meta_outside_dev(settings, baseten_env, app_env):
    h = env_harness(settings, app_env)
    h.sdk("baseten").script.extend([api_error(503)] * 3)
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert h.sdk("meta_api").calls == []


@pytest.mark.parametrize("app_env", ["demo", "prod"])
def test_unknown_provider_is_refused_outside_dev(settings, monkeypatch, app_env):
    monkeypatch.setenv("REASONING_PROVIDER", "meta_api")  # not an open-weight provider
    h = env_harness(settings, app_env)
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert h.sdk("meta_api").calls == []


def test_baseten_falls_back_to_our_glimmer_and_says_so(settings, baseten_env, monkeypatch):
    monkeypatch.setenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
    h = env_harness(settings, "demo")
    h.sdk("baseten").script.extend([api_error(503)] * 3)
    h.sdk("glimmer").script.append(chat_result(text=json.dumps(SCENE)))
    _, call = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert call.provider == "glimmer" and call.model == "muse-glimmer-30b"
    assert len(call.fallbacks) == 1 and "muse-spark-open-1026 on baseten failed" in call.fallbacks[0]
    assert h.sdk("meta_api").calls == []


def test_baseten_then_glimmer_both_down_gives_the_busy_error(settings, baseten_env, monkeypatch):
    monkeypatch.setenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
    h = env_harness(settings, "demo")
    h.sdk("baseten").script.extend([api_error(503)] * 3)
    h.sdk("glimmer").script.extend([api_error(503)] * 3)
    with pytest.raises(OpenWeightUnavailable, match="baseten.*glimmer"):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert h.sdk("meta_api").calls == []


def test_bad_request_does_not_fall_back_silently_to_a_retry(settings, baseten_env):
    """A 400 is our fault: it is never re-sent unchanged (and with no Glimmer server there is no fallback)."""
    h = Harness(settings)
    h.sdk("baseten").script.append(api_error(400))
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert len(h.sdk("baseten").calls) == 1


def test_missing_baseten_price_refuses_the_call(settings, baseten_env):
    path = settings.config_dir / "prices.yaml"
    prices = yaml.safe_load(path.read_text())
    del prices["baseten"]
    path.write_text(yaml.safe_dump(prices))
    h = Harness(settings)
    with pytest.raises(PriceMissing):
        h.client.open_weight_chat(messages=[{"role": "user", "content": "hi"}], run_id="r", provider="baseten")
    assert h.sdk("baseten").calls == []


def test_baseten_cost_uses_cached_and_reasoning_tokens(settings, baseten_env):
    """Real usage from the first test call (Oct 29): 169 in, 128 cached, 464 out of which 450 reasoning."""
    h = Harness(settings)
    resp = chat_result(text="2.01 s")
    resp.usage = SimpleNamespace(prompt_tokens=169, completion_tokens=464,
                                 prompt_tokens_details=SimpleNamespace(cached_tokens=128),
                                 completion_tokens_details=SimpleNamespace(reasoning_tokens=450))
    h.sdk("baseten").script.append(resp)
    r = h.client.open_weight_chat(messages=[{"role": "user", "content": "q"}], run_id="r", provider="baseten")
    expected = (41 * 1.25 + 128 * 0.15 + 464 * 4.25) / 1e6
    assert r.cost_usd == pytest.approx(expected)
    assert r.usage.reasoning_tokens == 450


def test_baseten_answers_are_cached(settings, baseten_env):
    h = Harness(settings)
    h.sdk("baseten").script.append(chat_result(text=json.dumps(SCENE)))
    ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    _, again = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r2")
    assert again.cached and again.cost_usd == 0 and len(h.sdk("baseten").calls) == 1


def test_baseten_budget_cap_blocks_before_calling(settings, baseten_env):
    path = settings.config_dir / "budget.yaml"
    budget = yaml.safe_load(path.read_text())
    budget["baseten"] = {"hard_cap_usd": 0.001, "daily_cap_usd": 0.001}
    path.write_text(yaml.safe_dump(budget))
    h = Harness(settings)
    from physicslens.costguard import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        h.client.open_weight_chat(messages=[{"role": "user", "content": "hi"}], run_id="r", provider="baseten")
    assert h.sdk("baseten").calls == []


# ---------- reasoning formats ----------

def test_reasoning_from_reasoning_content_field():
    """Baseten's documented format (and what the first real call returned)."""
    resp = chat_result(text="2.01 s")
    parsed = _parse_chat(resp)
    assert parsed.text == "2.01 s" and parsed.reasoning == "mock reasoning"


def test_reasoning_from_reasoning_field():
    resp = chat_result(text='{"a": 1}')
    resp.choices[0].message = SimpleNamespace(content='{"a": 1}', reasoning_content=None, reasoning="thought")
    assert _parse_chat(resp).reasoning == "thought"


def test_reasoning_inline_think_tags_are_split_off():
    text, reasoning = split_reasoning('<think>angle is small, so linear</think>\n{"a": 1}', None)
    assert text == '{"a": 1}' and reasoning == "angle is small, so linear"


def test_no_reasoning_and_no_tags_is_left_alone():
    assert split_reasoning('  {"a": 1} ', None) == ('{"a": 1}', None)
    # An unclosed tag is not guessed at: the validator sees it and the one retry handles it.
    assert split_reasoning("<think>never closed", None) == ("<think>never closed", None)


def test_health_names_the_reasoning_model(monkeypatch):
    """The site's 'server online' badge and the uptime check read this; it never exposes keys or quotas."""
    from fastapi.testclient import TestClient
    from physicslens import app as app_module
    client = TestClient(app_module.app)
    monkeypatch.setenv("REASONING_PROVIDER", "baseten")
    assert client.get("/api/health").json()["reasoning"] == {"provider": "baseten", "model": BASETEN_MODEL}
    monkeypatch.delenv("REASONING_PROVIDER")
    assert client.get("/api/health").json()["reasoning"] is None


@pytest.mark.parametrize(("task", "effort", "max_tokens"), [("hypotheses", "high", 16000), ("agent", "medium", 12000),
                                                         ("scene", "low", 8000), ("locate", "medium", 12000),
                                                         ("whatif", "low", 8000)])
def test_effort_and_output_budget_follow_the_step(settings, baseten_env, task, effort, max_tokens):
    """Hard decisions think longer and get room for it; short conversions stay cheap (whatif: 4 x 2000 answer)."""
    h = Harness(settings)
    h.sdk("baseten").script.append(chat_result(text=json.dumps(SCENE)))
    ask_json(h.client, task=task, system="s", text="t", schema_model=Scene, run_id="r")
    sent = h.sdk("baseten").calls[0]
    assert sent["extra_body"] == {"reasoning_effort": effort} and sent["max_tokens"] == max_tokens


def test_unlisted_step_uses_the_low_default(settings, baseten_env):
    h = Harness(settings)
    h.sdk("baseten").script.append(chat_result(text=json.dumps(SCENE)))
    ask_json(h.client, task="something_new", system="s", text="t", schema_model=Scene, run_id="r")
    assert h.sdk("baseten").calls[0]["extra_body"] == {"reasoning_effort": "low"}
