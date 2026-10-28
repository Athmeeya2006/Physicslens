import threading
from pathlib import Path

import pytest
import yaml
from physicslens.costguard import (
    BudgetExceeded,
    CircuitOpen,
    CostGuardClient,
    InvalidRequest,
    KillSwitchActive,
    ModelNotAllowed,
    OutputTruncated,
    PriceMissing,
    ProviderCallFailed,
    Usage,
)
from physicslens.costguard.client import openai_sdk_factory
from physicslens.costguard.guard import BUSY_MESSAGE
from physicslens.costguard.ledger import Ledger
from physicslens.costguard.mock import api_error, responses_result, sam_result
from physicslens.costguard.report import render
from physicslens.settings import Settings

from .conftest import Harness

BACKEND = Path(__file__).resolve().parents[1] / "backend" / "physicslens"


def text_call(client: CostGuardClient, run_id: str = "run-1", **kw):
    kw.setdefault("input", "hello")
    kw.setdefault("model", "muse-spark-1.3")
    return client.meta_text(run_id=run_id, **kw)


def statuses(client: CostGuardClient) -> list[str]:
    return [r["status"] for r in client.ledger.rows()]


# ---------- kill switch, allowlist, prices ----------

def test_kill_file_blocks_before_any_request(harness, settings):
    settings.kill_file.touch()
    with pytest.raises(KillSwitchActive):
        text_call(harness.client)
    assert harness.sdk().calls == []
    assert harness.client.ledger.rows() == []


def test_kill_env_blocks(harness, monkeypatch):
    monkeypatch.setenv("PHYSICSLENS_KILL", "1")
    with pytest.raises(KillSwitchActive):
        text_call(harness.client)
    assert harness.sdk().calls == []


def test_allowlist_blocks_unlisted_models(harness):
    with pytest.raises(ModelNotAllowed):
        text_call(harness.client, model="muse-image-1.0")
    assert harness.sdk().calls == []


def test_wonder_needs_its_own_daily_cap(harness):
    # Approved for controller code only, and only with a model cap of its own in budget.yaml.
    with pytest.raises(ModelNotAllowed, match="model_daily_caps_usd"):
        text_call(harness.client, model="muse-wonder-1.0")
    assert harness.sdk().calls == []


def test_wonder_cap_and_audience(settings):
    path = settings.config_dir / "budget.yaml"
    budget = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**budget, "model_daily_caps_usd": {"muse-wonder-1.0": 0.05}}))
    h = Harness(settings)
    with pytest.raises(ModelNotAllowed, match="public"):
        text_call(h.client, model="muse-wonder-1.0", audience="public")
    # 16 output tokens at the Wonder output price fit; 8000 do not fit a 0.05 daily cap for this one model.
    text_call(h.client, model="muse-wonder-1.0", max_output_tokens=16)
    with pytest.raises(BudgetExceeded, match="muse-wonder-1.0 daily budget"):
        text_call(h.client, model="muse-wonder-1.0", max_output_tokens=8000, run_id="run-2")
    assert len(h.sdk().calls) == 1


def test_model_must_match_provider(harness):
    with pytest.raises(ModelNotAllowed):
        text_call(harness.client, model="muse-glimmer-30b")


def test_missing_price_refuses_call(settings, monkeypatch):
    path = settings.config_dir / "prices.yaml"
    prices = yaml.safe_load(path.read_text())
    del prices["glimmer"]
    path.write_text(yaml.safe_dump(prices))
    monkeypatch.setenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
    h = Harness(settings)
    with pytest.raises(PriceMissing):
        h.client.glimmer_chat(messages=[{"role": "user", "content": "hi"}], run_id="r")
    assert h.sdk("glimmer").calls == []


def test_contributor_refused_outside_dev(settings):
    demo = Settings(app_env="demo", repo_root=settings.repo_root, config_dir=settings.config_dir, var_dir=settings.var_dir)
    h = Harness(demo)
    with pytest.raises(ModelNotAllowed):
        text_call(h.client, model="muse-spark-1.3-contributor")


def test_real_sdk_client_has_no_hidden_retries(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-not-a-real-key")
    sdk = openai_sdk_factory("meta_api", "https://api.meta.ai/v1", "MODEL_API_KEY", 60)
    assert sdk.max_retries == 0


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(Exception, match="MODEL_API_KEY is not set"):
        openai_sdk_factory("meta_api", "https://api.meta.ai/v1", "MODEL_API_KEY", 60)


def test_only_client_constructs_sdk():
    offenders = [
        p.relative_to(BACKEND) for p in BACKEND.rglob("*.py")
        if p.name != "client.py" and ("OpenAI(" in p.read_text() or "import websockets" in p.read_text())
    ]
    assert offenders == []


# ---------- request rules ----------

@pytest.mark.parametrize("effort", ["high", "xhigh", "max", "none"])
def test_expensive_meta_effort_refused(harness, effort):
    with pytest.raises(InvalidRequest):
        text_call(harness.client, effort=effort)


@pytest.mark.parametrize("strength", ["high", "xhigh", "minimal"])
def test_expensive_glimmer_strength_refused(harness, strength):
    with pytest.raises(InvalidRequest):
        harness.client.glimmer_chat(messages=[], run_id="r", strength=strength)


def test_meta_request_shape(harness):
    text_call(harness.client, prompt_cache_key="physicslens-test-v1", schema={"type": "object"}, schema_name="x")
    call = harness.sdk().calls[0]
    assert call["store"] is False
    assert call["reasoning"] == {"effort": "minimal"}
    assert call["max_output_tokens"] == 2000
    assert call["prompt_cache_key"] == "physicslens-test-v1"
    assert call["text"]["format"]["strict"] is True
    assert "temperature" not in call and "top_p" not in call


# ---------- budget ----------

def test_cost_recorded_from_usage(harness):
    r = text_call(harness.client)
    assert r.cost_usd == pytest.approx((100 * 1.25 + 50 * 4.25) / 1e6)
    row = harness.client.ledger.rows()[0]
    assert row["status"] == "committed" and row["output_tokens"] == 50 and row["reasoning_tokens"] == 10


def test_budget_exceeded_raises(harness):
    with pytest.raises(BudgetExceeded):
        harness.client.guard.check_and_reserve("meta_api", "muse-spark-1.3", 0.60, "run-x")


def test_daily_and_total_caps(harness):
    g = harness.client.guard
    g.check_and_reserve("meta_api", "muse-spark-1.3", 0.15, "a").commit(0.15)
    g.check_and_reserve("meta_api", "muse-spark-1.3", 0.15, "b").commit(0.15)
    g.check_and_reserve("meta_api", "muse-spark-1.3", 0.15, "c").commit(0.15)
    with pytest.raises(BudgetExceeded, match="daily"):
        g.check_and_reserve("meta_api", "muse-spark-1.3", 0.10, "d")


def test_per_run_cap(harness):
    g = harness.client.guard
    g.check_and_reserve("meta_api", "muse-spark-1.3", 0.15, "same-run").commit(0.15)
    with pytest.raises(BudgetExceeded, match="per-run"):
        g.check_and_reserve("meta_api", "muse-spark-1.3", 0.10, "same-run")


def test_judge_cap_shows_polite_message(harness):
    g = harness.client.guard
    g.check_and_reserve("meta_api", "muse-spark-1.3", 0.15, "j1", audience="judge").commit(0.15)
    g.check_and_reserve("meta_api", "sam-3.1", 0.15, "j2", audience="judge").commit(0.15)
    with pytest.raises(BudgetExceeded) as e:
        g.check_and_reserve("meta_api", "muse-spark-1.3", 0.05, "j3", audience="judge")
    assert str(e.value) == BUSY_MESSAGE


def test_concurrent_reservations_cannot_overshoot(harness):
    g = harness.client.guard
    ok, refused = [], []
    barrier = threading.Barrier(20)

    def worker(i):
        barrier.wait()
        try:
            ok.append(g.check_and_reserve("meta_api", "muse-spark-1.3", 0.04, f"run-{i}"))
        except BudgetExceeded:
            refused.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 12  # 12 * 0.04 = 0.48 <= 0.50 daily cap
    assert len(refused) == 8


def test_failed_call_releases_reservation(harness):
    harness.sdk().script.append(api_error(400))
    with pytest.raises(ProviderCallFailed):
        text_call(harness.client)
    assert statuses(harness.client) == ["released"]


def test_missing_usage_charges_estimate(harness):
    harness.sdk().script.append(responses_result(usage=False))
    r = text_call(harness.client)
    assert r.cost_usd > 0
    assert harness.client.ledger.rows()[0]["note"] == "usage missing; charged estimate"


def test_truncated_output_is_billed_and_raises(harness):
    harness.sdk().script.append(responses_result(text="", incomplete="max_output_tokens"))
    with pytest.raises(OutputTruncated):
        text_call(harness.client)
    assert statuses(harness.client) == ["committed"]
    harness.sdk().script.clear()
    text_call(harness.client)  # not cached: a second call reaches the provider
    assert len(harness.sdk().calls) == 2


def test_ledger_persists_across_instances(harness, settings):
    text_call(harness.client)
    assert len(Ledger(settings.ledger_path).rows()) == 1


# ---------- cache ----------

def test_cache_hit_records_zero(harness):
    first = text_call(harness.client)
    second = text_call(harness.client)
    assert len(harness.sdk().calls) == 1
    assert second.cached and second.cost_usd == 0 and second.text == first.text
    rows = harness.client.ledger.rows()
    assert rows[1]["status"] == "cache_hit" and rows[1]["est_cost_usd"] == 0


# ---------- retries and circuit breaker ----------

def test_429_respects_retry_after_then_succeeds(harness):
    harness.sdk().script.extend([api_error(429, {"retry-after": "3"}), api_error(429)])
    text_call(harness.client)
    assert harness.sleeps[0] == 3.0
    assert 1.0 <= harness.sleeps[1] <= 1.25  # second retry: 500 ms -> 1 s
    assert len(harness.sdk().calls) == 3


def test_429_max_three_retries(harness):
    harness.sdk().script.extend([api_error(429)] * 5)
    with pytest.raises(ProviderCallFailed):
        text_call(harness.client)
    assert len(harness.sdk().calls) == 4


def test_400_never_retried(harness):
    harness.sdk().script.append(api_error(400))
    with pytest.raises(ProviderCallFailed):
        text_call(harness.client)
    assert len(harness.sdk().calls) == 1


def test_503_retried_twice(harness):
    harness.sdk().script.extend([api_error(503)] * 3)
    with pytest.raises(ProviderCallFailed):
        text_call(harness.client)
    assert len(harness.sdk().calls) == 3


def test_circuit_breaker_opens_after_five_failures(harness):
    harness.sdk().script.extend([api_error(502)] * 6)
    for run in ("a", "b"):
        with pytest.raises(ProviderCallFailed):
            text_call(harness.client, run_id=run, input=run)
    with pytest.raises(CircuitOpen):
        text_call(harness.client, run_id="c", input="c")


# ---------- SAM ----------

def sam_call(client: CostGuardClient, **kw):
    kw.setdefault("est_frames", 60)
    return client.sam_video(prompt="red ball", video_hash="abc", file_id="file-1", run_id="r", **kw)


def test_sam_bills_reported_frames_with_50_minimum(harness):
    harness.sdk().script.append(sam_result(frames=20))
    r = sam_call(harness.client)
    assert r.usage.video_frames == 20
    assert r.cost_usd == pytest.approx(50 * 0.0002)


def test_sam_retries_once_and_failure_is_billed(harness):
    harness.sdk().script.extend([api_error(503), api_error(503)])
    with pytest.raises(ProviderCallFailed):
        sam_call(harness.client)
    assert len(harness.sdk().calls) == 2
    row = harness.client.ledger.rows()[0]
    assert row["status"] == "failed_billed" and row["est_cost_usd"] == pytest.approx(60 * 0.0002)


def test_sam_metadata_must_be_strings(harness):
    with pytest.raises(InvalidRequest):
        sam_call(harness.client, metadata={"include_confidence": True})


def test_sam_cache_keyed_by_video_hash_not_file_id(harness):
    sam_call(harness.client)
    r = harness.client.sam_video(prompt="red ball", video_hash="abc", file_id="file-2", est_frames=60, run_id="r2")
    assert r.cached and len(harness.sdk().calls) == 1


def test_upload_reused_by_hash_with_expiry(harness, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video")
    a = harness.client.upload_video(clip)
    b = harness.client.upload_video(clip)
    assert a == b
    files = [c for c in harness.sdk().calls if c["kind"] == "files"]
    assert len(files) == 1
    assert files[0]["expires_after"] == {"anchor": "created_at", "seconds": 86400}


# ---------- voice, pricing, limiter ----------

def test_voice_bills_whole_seconds(harness):
    with harness.client.voice_session(run_id="v", max_seconds=60) as usage:
        usage.audio_seconds = 10.2
    row = harness.client.ledger.rows()[0]
    assert row["est_cost_usd"] == pytest.approx(11 * 0.18 / 3600)


def test_voice_concurrency_cap(harness):
    c = harness.client
    with (
        c.voice_session(run_id="a", max_seconds=5),
        c.voice_session(run_id="b", max_seconds=5),
        pytest.raises(Exception, match="too many concurrent"),
        c.voice_session(run_id="c", max_seconds=5),
    ):
        pass


def test_reasoning_reported_separately_is_billed(harness):
    p = harness.client.pricing
    assert p.cost("meta_api", "muse-spark-1.3", Usage(output_tokens=10, reasoning_tokens=100)) == pytest.approx(110 * 4.25 / 1e6)


def test_rate_limiter_paces_requests():
    from physicslens.costguard.limiter import TokenBucket

    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    bucket = TokenBucket(30, clock=lambda: now[0], sleep=sleep)
    for _ in range(7):
        bucket.acquire(timeout=60)
    assert now[0] == pytest.approx(4.0)  # burst of 5, then one every 2 s at 30 RPM


def test_low_remaining_header_slows_down(harness):
    harness.sdk().headers = {"x-ratelimit-remaining-requests": "5", "x-ratelimit-limit-requests": "3000"}
    text_call(harness.client)
    assert harness.client.limiter.bucket("muse-spark-1.3").paused_until > 0


# ---------- report ----------

def test_report_renders(harness):
    text_call(harness.client)
    out = render(harness.client.ledger, harness.client.guard)
    assert "muse-spark-1.3" in out and "dashboard" in out


def test_cached_sam_result_never_uploads(harness, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake video bytes")
    from physicslens.costguard.cache import file_digest
    kw = {"prompt": "red ball", "video_hash": file_digest(clip), "video_path": clip, "est_frames": 60, "run_id": "r"}
    harness.client.sam_video(**kw)
    uploads = [c for c in harness.sdk().calls if c["kind"] == "files"]
    assert len(uploads) == 1
    r = harness.client.sam_video(**kw)
    assert r.cached
    assert len([c for c in harness.sdk().calls if c["kind"] == "files"]) == 1
