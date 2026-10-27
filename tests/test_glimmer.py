"""Self-hosted Muse Glimmer: the open-weight reasoning path, and the cache-only mode a GPU notebook uses."""

import json

import pytest
from physicslens.agent.llm import OpenWeightUnavailable, ask_json
from physicslens.agent.schemas import Scene
from physicslens.costguard import CacheOnlyMiss
from physicslens.costguard.mock import api_error, chat_result, responses_result

from .conftest import Harness
from .test_pipeline import SCENE


@pytest.fixture
def glimmer_env(monkeypatch):
    monkeypatch.setenv("REASONING_PROVIDER", "glimmer")
    monkeypatch.setenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("GLIMMER_API_KEY", "test-not-a-real-key")


def test_glimmer_used_when_selected(settings, glimmer_env):
    h = Harness(settings)
    h.sdk("glimmer").script.append(chat_result(text=json.dumps(SCENE)))
    scene, call = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert call.provider == "glimmer" and call.model == "muse-glimmer-30b" and call.cost_usd == 0.0
    assert scene.moving_object == "red ball"
    sent = h.sdk("glimmer").calls[0]
    # The chat template defaults to high reasoning; low is always sent explicitly.
    assert sent["extra_body"] == {"chat_template_kwargs": {"reasoning_strength": "low"}}
    assert "temperature" not in sent
    assert h.sdk("meta_api").calls == []


def test_glimmer_failure_never_falls_back_even_in_dev(settings, glimmer_env):
    h = Harness(settings)
    assert settings.app_env == "dev"
    h.sdk("glimmer").script.extend([api_error(503)] * 3)
    h.sdk("meta_api").script.append(responses_result(text=json.dumps(SCENE)))
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert h.sdk("meta_api").calls == []


def test_glimmer_answers_are_cached(settings, glimmer_env):
    h = Harness(settings)
    h.sdk("glimmer").script.append(chat_result(text=json.dumps(SCENE)))
    ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    _, again = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r2")
    assert again.cached and len(h.sdk("glimmer").calls) == 1


def test_glimmer_needs_a_server_address(settings, monkeypatch):
    monkeypatch.setenv("REASONING_PROVIDER", "glimmer")
    monkeypatch.delenv("GLIMMER_BASE_URL", raising=False)
    from physicslens.costguard.allowlist import Allowlist
    from physicslens.costguard.errors import CostGuardError
    from physicslens.settings import load_yaml
    with pytest.raises(CostGuardError, match="GLIMMER_BASE_URL"):
        Allowlist(load_yaml(settings.config_dir, "models.yaml")).base_url("glimmer")


def test_cache_only_provider_replays_hits_and_refuses_misses(settings, monkeypatch):
    h = Harness(settings)
    h.sdk().script.append(responses_result(text="first"))
    first = h.client.meta_text(input="hello", run_id="r")
    assert first.text == "first"

    monkeypatch.setenv("PHYSICSLENS_CACHE_ONLY", "meta_api")
    replay = Harness(settings)
    assert replay.client.meta_text(input="hello", run_id="r2").cached
    with pytest.raises(CacheOnlyMiss):
        replay.client.meta_text(input="something new", run_id="r3")
    assert replay.sdk().calls == []
    # Nothing was reserved or billed for the miss.
    assert replay.client.guard.summary()["meta_api"]["spent_usd"] == pytest.approx(first.cost_usd, abs=1e-6)


def test_cache_only_machine_needs_no_meta_key(settings, monkeypatch):
    """The GPU notebook replays SAM from the cache with the real SDK factory and no key at all."""
    from physicslens.costguard import CostGuardClient
    from physicslens.costguard.client import openai_sdk_factory
    from physicslens.costguard.mock import sam_result

    h = Harness(settings)
    h.sdk().script.append(sam_result(frames=60, text="<0f>1<|box;x1=1;y1=1;x2=5;y2=5;w=64;h=64|>"))
    args = {"prompt": "p", "video_hash": "abc", "est_frames": 60, "video_url": "data:video/mp4;base64,AAAA"}
    first = h.client.sam_video(run_id="r", **args)

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("PHYSICSLENS_CACHE_ONLY", "meta_api")
    gpu_side = CostGuardClient(settings, sdk_factory=openai_sdk_factory)
    replay = gpu_side.sam_video(run_id="r2", **args)
    assert replay.cached and replay.text == first.text
    with pytest.raises(CacheOnlyMiss):
        gpu_side.sam_video(run_id="r3", **{**args, "prompt": "other"})
