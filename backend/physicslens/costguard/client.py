"""The ONLY module that calls provider SDKs. Every paid call follows the same order:
kill switch -> allowlist -> price exists -> budget reserve -> rate limiter -> cache -> request -> usage -> commit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
import websockets
from openai import OpenAI
from websockets.exceptions import ConnectionClosed

from ..settings import (
    GLIMMER_MODEL,
    META_CONTRIBUTOR_MODEL,
    SAM_MODEL,
    VOICE_MODEL,
    Settings,
    load_budget,
    load_yaml,
)
from .allowlist import Allowlist
from .cache import ResponseCache, file_digest, make_key
from .errors import (
    CacheOnlyMiss,
    CostGuardError,
    InvalidRequest,
    KillSwitchActive,
    ModelNotAllowed,
    OutputTruncated,
    ProviderCallFailed,
)
from .guard import Guard, Reservation
from .ledger import Ledger
from .limiter import CircuitBreaker, RateLimiter
from .pricing import Pricing, Usage

log = logging.getLogger("physicslens.costguard")

META_EFFORTS = ("minimal", "low", "medium")  # never high/xhigh/max; "none" is a 400
GLIMMER_STRENGTHS = ("low", "medium")  # the chat template defaults to high, so it is always sent
VOICE_URL = "wss://api.meta.ai/v1/asr/realtime"
VOICE_BYTES_PER_SECOND = 24_000 * 2  # 24 kHz, 16-bit, mono
VOICE_MAX_AHEAD_S = 5.0
IMAGE_TOKEN_ESTIMATE = 1600  # ~1,300-1,500 for a 1280 px image, rounded up
CHARS_PER_TOKEN = 3  # deliberately pessimistic


@dataclass
class CallResult:
    model: str
    text: str
    usage: Usage
    cost_usd: float
    cached: bool = False
    reasoning: str | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_cache(self) -> dict[str, Any]:
        return {"model": self.model, "text": self.text, "usage": self.usage.as_dict(),
                "reasoning": self.reasoning, "finish_reason": self.finish_reason, "raw": self.raw}

    @classmethod
    def from_cache(cls, value: dict[str, Any]) -> CallResult:
        return cls(model=value["model"], text=value["text"], usage=Usage(**value["usage"]), cost_usd=0.0,
                   cached=True, reasoning=value.get("reasoning"), finish_reason=value.get("finish_reason"),
                   raw=value.get("raw") or {})


@dataclass
class _Parsed:
    text: str
    usage: Usage | None
    reasoning: str | None = None
    finish_reason: str | None = None
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RetryPolicy:
    on_429: int
    on_transient: int


SdkFactory = Callable[[str, str, str, float], Any]


def openai_sdk_factory(provider: str, base_url: str, api_key_env: str, timeout: float) -> OpenAI:
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise CostGuardError(f"{api_key_env} is not set. Add it to .env (never commit it).")
    # max_retries=0: the SDK's hidden retries would bypass our budget and retry rules.
    return OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=timeout)


class CostGuardClient:
    def __init__(
        self,
        settings: Settings,
        *,
        sdk_factory: SdkFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        voice_connect: Callable[..., Any] | None = None,
    ):
        self.settings = settings
        cfg = settings.config_dir
        self.limits = load_yaml(cfg, "limits.yaml")
        self.allowlist = Allowlist(load_yaml(cfg, "models.yaml"))
        media = self.limits.get("media") or {}
        self.pricing = Pricing(load_yaml(cfg, "prices.yaml"), sam_min_frames=int(media.get("sam_min_frames_per_request", 50)))
        self.ledger = Ledger(settings.ledger_path)
        self.guard = Guard(self.ledger, load_budget(cfg))
        self.cache = ResponseCache(settings.cache_path)
        self.limiter = RateLimiter(self.limits, clock=clock, sleep=sleep)
        cb = self.limits.get("circuit_breaker") or {}
        self.breaker = CircuitBreaker(int(cb.get("consecutive_failures", 5)), float(cb.get("pause_seconds", 120)), clock=clock)
        if sdk_factory is None:
            if settings.mock_models:
                from .mock import mock_sdk_factory
                sdk_factory = mock_sdk_factory
            else:
                sdk_factory = openai_sdk_factory
        self._sdk_factory = sdk_factory
        if voice_connect is None:
            if settings.mock_models:
                from .mock import mock_voice_connect
                voice_connect = mock_voice_connect
            else:
                voice_connect = websockets.connect
        self._voice_connect = voice_connect
        self._sdks: dict[str, Any] = {}
        self._sleep = sleep
        timeouts = self.limits.get("timeouts_seconds") or {}
        self.llm_timeout = float(timeouts.get("llm", 60))
        self.sam_timeout = float(timeouts.get("sam_clip", 120))
        self.glimmer_timeout = float(timeouts.get("glimmer", 900))
        retries = self.limits.get("retries") or {}
        self._backoff_start = float(retries.get("backoff_start_ms", 500)) / 1000
        self._llm_policy = _RetryPolicy(int(retries.get("on_429_max", 3)), int(retries.get("on_transient_5xx_max", 2)))
        # A failed SAM stream may still be billed, so it gets one transient retry; 429s are never billed.
        self._sam_policy = _RetryPolicy(int(retries.get("on_429_max", 3)), int(retries.get("sam_max", 1)))
        out = self.limits.get("output_token_limits") or {}
        self.meta_max_output_default = int(out.get("meta_max_output_tokens_default", 2000))
        self.glimmer_max_tokens_default = int(out.get("glimmer_max_tokens_default", 8000))
        # Providers listed in PHYSICSLENS_CACHE_ONLY answer from the cache or not at all. A GPU server
        # running the open-weight model replays SAM results this way and never holds a Meta key.
        self.cache_only = {p.strip() for p in os.environ.get("PHYSICSLENS_CACHE_ONLY", "").split(",") if p.strip()}

    # ---------- preflight checks ----------

    def check_kill_switch(self) -> None:
        if self.settings.kill_file.exists() or os.environ.get("PHYSICSLENS_KILL", "0").strip() == "1":
            raise KillSwitchActive("Kill switch is ON (.KILL file or PHYSICSLENS_KILL=1): all paid calls are blocked.")

    def _check_model(self, model: str, expected_provider: str) -> str:
        provider = self.allowlist.provider_for(model)
        if provider != expected_provider:
            raise ModelNotAllowed(f"{model!r} belongs to {provider}, not {expected_provider}.")
        if model == META_CONTRIBUTOR_MODEL and self.settings.app_env != "dev":
            raise ModelNotAllowed(f"{model} is only allowed in APP_ENV=dev.")
        self.pricing.entry(provider, model)
        return provider

    def _sdk(self, provider: str, timeout: float) -> Any:
        if provider not in self._sdks:
            self._sdks[provider] = self._sdk_factory(
                provider, self.allowlist.base_url(provider), self.allowlist.api_key_env(provider), timeout
            )
        return self._sdks[provider]

    # ---------- shared execution path ----------

    def _execute(
        self,
        *,
        provider: str,
        model: str,
        run_id: str,
        audience: str,
        est_cost: float,
        cache_key: str | None,
        request: Callable[[], tuple[Any, Any]],
        parse: Callable[[Any], _Parsed],
        policy: _RetryPolicy,
        billed_on_failure: bool = False,
        video_requests: int = 0,
        run_cap_usd: float | None = None,
    ) -> CallResult:
        self.check_kill_switch()
        self._check_model(model, provider)
        res = self.guard.check_and_reserve(provider, model, est_cost, run_id, audience, run_cap_usd)
        try:
            self.limiter.acquire(model, timeout=self.llm_timeout)
            if cache_key:
                hit = self.cache.get(cache_key)
                if hit is not None:
                    res.commit(0.0, status="cache_hit")
                    log.info("paid-call provider=%s model=%s run=%s cache_hit cost=$0", provider, model, run_id)
                    return CallResult.from_cache(hit)
            if provider in self.cache_only:
                raise CacheOnlyMiss(f"{provider}/{model}: not in the cache, and {provider} is cache-only here.")
            self.breaker.check(provider)
            response, headers = self._with_retries(provider, model, request, policy)
            if headers:
                self.limiter.observe_headers(model, headers)
            parsed = parse(response)
            return self._settle(res, parsed, provider, model, run_id, est_cost, cache_key, video_requests)
        except BaseException as e:
            if not res.settled:
                if billed_on_failure and not isinstance(e, KillSwitchActive):
                    res.commit(est_cost, status="failed_billed", note=f"{type(e).__name__}: charged estimate")
                else:
                    res.release()
            raise

    def _settle(self, res: Reservation, parsed: _Parsed, provider: str, model: str, run_id: str,
                est_cost: float, cache_key: str | None, video_requests: int) -> CallResult:
        note = None
        if parsed.usage is None:
            log.warning("%s/%s returned no usage; charging the reservation estimate $%.5f.", provider, model, est_cost)
            cost, usage, note = est_cost, Usage(), "usage missing; charged estimate"
        else:
            usage = parsed.usage
            cost = self.pricing.cost(provider, model, usage, video_requests=video_requests)
        res.commit(cost, usage, note=note)
        log.info(
            "paid-call provider=%s model=%s run=%s in=%d cached=%d out=%d reasoning=%d frames=%d cost=$%.5f",
            provider, model, run_id, usage.input_tokens, usage.cached_tokens, usage.output_tokens,
            usage.reasoning_tokens, usage.video_frames, cost,
        )
        result = CallResult(model=model, text=parsed.text, usage=usage, cost_usd=cost, reasoning=parsed.reasoning,
                            finish_reason=parsed.finish_reason, raw=parsed.raw)
        if parsed.truncated:
            raise OutputTruncated(
                f"{model} hit its output-token limit (finish={parsed.finish_reason}); the call cost ${cost:.5f}. "
                "Raise the output limit for this prompt rather than retrying unchanged."
            )
        if cache_key:
            self.cache.put(cache_key, model, result.to_cache())
        return result

    def _backoff(self, attempt: int) -> float:
        delay = self._backoff_start * 2**attempt
        return delay + random.uniform(0, delay * 0.25)

    def _with_retries(self, provider: str, model: str, request: Callable[[], tuple[Any, Any]], policy: _RetryPolicy) -> tuple[Any, Any]:
        n429 = ntransient = 0
        while True:
            self.check_kill_switch()
            try:
                out = request()
                self.breaker.record_success(provider)
                return out
            except openai.APIStatusError as e:
                code = e.status_code
                if code == 429:
                    if n429 >= policy.on_429:
                        raise ProviderCallFailed(f"{provider}/{model}: rate limited (429) after {n429} retries.", 429) from e
                    wait = _retry_after(e) or self._backoff(n429)
                    log.warning("%s/%s 429; waiting %.1f s (retry %d/%d).", provider, model, wait, n429 + 1, policy.on_429)
                    self.limiter.pause(model, wait)
                    self._sleep(wait)
                    n429 += 1
                    continue
                if code < 500:
                    # 4xx means our request is wrong; sending it again unchanged would just pay twice.
                    raise ProviderCallFailed(f"{provider}/{model}: HTTP {code}: {_error_message(e)}", code) from e
                self.breaker.record_failure(provider)
                if ntransient >= policy.on_transient:
                    raise ProviderCallFailed(f"{provider}/{model}: HTTP {code} after {ntransient} retries.", code) from e
            except openai.APIConnectionError as e:  # includes timeouts and dropped streams
                self.breaker.record_failure(provider)
                if ntransient >= policy.on_transient:
                    raise ProviderCallFailed(f"{provider}/{model}: {type(e).__name__} after {ntransient} retries.") from e
            wait = self._backoff(ntransient)
            log.warning("%s/%s transient failure; retrying in %.1f s (%d/%d).", provider, model, wait, ntransient + 1, policy.on_transient)
            self._sleep(wait)
            ntransient += 1

    # ---------- Meta Model API: text (Responses API) ----------

    def meta_text(
        self,
        *,
        input: str | list[dict[str, Any]],
        run_id: str,
        model: str | None = None,
        instructions: str | None = None,
        schema: dict[str, Any] | None = None,
        schema_name: str = "output",
        effort: str = "minimal",
        max_output_tokens: int | None = None,
        prompt_cache_key: str | None = None,
        audience: str = "dev",
        preflight_tokens: bool = False,
        use_cache: bool = True,
        run_cap_usd: float | None = None,
    ) -> CallResult:
        model = model or self.settings.meta_text_model()
        if effort not in META_EFFORTS:
            raise InvalidRequest(f"Meta reasoning effort must be one of {META_EFFORTS}, got {effort!r}.")
        max_out = max_output_tokens or self.meta_max_output_default
        if max_out < 16:
            raise InvalidRequest("max_output_tokens must be at least 16.")
        provider = "meta_api"
        self.check_kill_switch()
        self._check_model(model, provider)

        body: dict[str, Any] = {"model": model, "input": input, "store": False,
                                "reasoning": {"effort": effort}, "max_output_tokens": max_out}
        if instructions:
            body["instructions"] = instructions
        if schema is not None:
            body["text"] = {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}}
        if prompt_cache_key:
            body["prompt_cache_key"] = prompt_cache_key

        est_in = self.count_input_tokens(body) if preflight_tokens else _estimate_input_tokens([instructions, input])
        est_cost = self.pricing.worst_case_tokens(provider, model, est_in, max_out)

        def request() -> tuple[Any, Any]:
            # Created on first real request: a cache hit (or a cache-only machine) needs no key.
            sdk = self._sdk(provider, self.llm_timeout)
            raw = sdk.responses.with_raw_response.create(**body, timeout=self.llm_timeout)
            return raw.parse(), raw.headers

        return self._execute(
            provider=provider, model=model, run_id=run_id, audience=audience, est_cost=est_cost,
            cache_key=make_key(model, body) if use_cache else None, request=request,
            parse=_parse_responses, policy=self._llm_policy, run_cap_usd=run_cap_usd,
        )

    def count_input_tokens(self, body: dict[str, Any]) -> int:
        """Free preflight count; it slightly overestimates, which is the safe direction."""
        self.check_kill_switch()
        sdk = self._sdk("meta_api", self.llm_timeout)
        keys = ("model", "input", "instructions", "reasoning", "text")
        resp = sdk.responses.input_tokens.count(**{k: body[k] for k in keys if k in body})
        return int(resp.input_tokens)

    # ---------- Muse Glimmer: open weights on our own llama-server (Chat Completions) ----------

    def glimmer_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        run_id: str,
        strength: str = "low",
        max_tokens: int | None = None,
        audience: str = "dev",
        use_cache: bool = True,
        run_cap_usd: float | None = None,
    ) -> CallResult:
        if strength not in GLIMMER_STRENGTHS:
            raise InvalidRequest(f"Glimmer reasoning strength must be one of {GLIMMER_STRENGTHS}, got {strength!r}.")
        provider, model = "glimmer", GLIMMER_MODEL
        self.check_kill_switch()
        self._check_model(model, provider)
        max_tok = max_tokens or self.glimmer_max_tokens_default
        # Sampling stays at the model's defaults (temperature 1.0); temperature 0 makes it loop.
        body = {"model": model, "messages": messages, "max_tokens": max_tok,
                "chat_template_kwargs": {"reasoning_strength": strength}}
        est_cost = self.pricing.worst_case_tokens(provider, model, _estimate_input_tokens(messages), max_tok)

        def request() -> tuple[Any, Any]:
            sdk = self._sdk(provider, self.glimmer_timeout)
            raw = sdk.chat.completions.with_raw_response.create(
                model=model, messages=messages, max_tokens=max_tok, timeout=self.glimmer_timeout,
                extra_body={"chat_template_kwargs": body["chat_template_kwargs"]})
            return raw.parse(), raw.headers

        return self._execute(
            provider=provider, model=model, run_id=run_id, audience=audience, est_cost=est_cost,
            cache_key=make_key(model, body) if use_cache else None, request=request,
            parse=_parse_chat, policy=self._llm_policy, run_cap_usd=run_cap_usd,
        )

    # ---------- SAM 3.1 (video mode only) ----------

    def sam_video(
        self,
        *,
        prompt: str,
        video_hash: str,
        est_frames: int,
        run_id: str,
        file_id: str | None = None,
        video_url: str | None = None,
        video_path: Path | None = None,
        metadata: dict[str, str] | None = None,
        audience: str = "dev",
        run_cap_usd: float | None = None,
    ) -> CallResult:
        """video_path uploads lazily inside the request, so a cached result never uploads anything."""
        if sum(x is not None for x in (file_id, video_url, video_path)) != 1:
            raise InvalidRequest("Pass exactly one of file_id, video_url or video_path.")
        metadata = metadata or {}
        if not all(isinstance(v, str) for v in metadata.values()):
            raise InvalidRequest("SAM metadata values must all be strings.")
        provider, model = "meta_api", SAM_MODEL
        self.check_kill_switch()
        self._check_model(model, provider)
        est_cost = self.pricing.cost(provider, model, Usage(video_frames=est_frames), video_requests=1)

        def stream_once(video: dict[str, Any]) -> tuple[Any, Any]:
            sdk = self._sdk(provider, self.sam_timeout)
            sam_input = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, video]}]
            with sdk.responses.stream(model=model, input=sam_input, metadata=metadata, timeout=self.sam_timeout) as stream:
                return stream.get_final_response(), None

        def request() -> tuple[Any, Any]:
            if video_path is not None:
                uploaded_id, _ = self.upload_video(video_path)
                try:
                    return stream_once({"type": "input_video", "file_id": uploaded_id})
                finally:
                    # Uploaded clips are deleted as soon as SAM is done with them, not left to expire.
                    self.delete_file(uploaded_id, video_hash)
            if file_id is not None:
                return stream_once({"type": "input_video", "file_id": file_id})
            return stream_once({"type": "input_video", "video_url": video_url})

        # Keyed by the video's content hash, never the file_id: the same clip re-uploaded must still hit.
        cache_key = make_key(model, {"prompt": prompt, "metadata": metadata}, media=[video_hash])
        return self._execute(
            provider=provider, model=model, run_id=run_id, audience=audience, est_cost=est_cost,
            cache_key=cache_key, request=request, parse=_parse_sam, policy=self._sam_policy,
            billed_on_failure=True, video_requests=1, run_cap_usd=run_cap_usd,
        )

    # ---------- Files API ----------

    def upload_video(self, path: Path) -> tuple[str, str]:
        """Upload once per clip (reused by content hash until it expires). Returns (file_id, video_hash)."""
        self.check_kill_switch()
        video_hash = file_digest(path)
        key = f"file:{video_hash}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit["file_id"], video_hash
        ttl = int((self.limits.get("media") or {}).get("files_api_expires_seconds", 86400))
        sdk = self._sdk("meta_api", self.sam_timeout)
        with open(path, "rb") as f:
            uploaded = sdk.files.create(purpose="user_data", file=f, expires_after={"anchor": "created_at", "seconds": ttl})
        # Forget it 10 minutes before the provider does, so we never reuse an expired id.
        self.cache.put(key, "files", {"file_id": uploaded.id}, ttl_seconds=max(ttl - 600, 60))
        log.info("uploaded %s as %s (expires in %d s)", Path(path).name, uploaded.id, ttl)
        return uploaded.id, video_hash

    def delete_file(self, file_id: str, video_hash: str | None = None) -> None:
        sdk = self._sdk("meta_api", self.sam_timeout)
        sdk.files.delete(file_id)
        if video_hash:
            self.cache.put(f"file:{video_hash}", "files", {}, ttl_seconds=-1)

    def list_models(self, provider: str) -> list[str]:
        self.check_kill_switch()
        sdk = self._sdk(provider, self.llm_timeout)
        return sorted(m.id for m in sdk.models.list())

    # ---------- Voice: Muse Voice Transcribe realtime ----------

    @contextmanager
    def voice_session(self, *, run_id: str, max_seconds: float, audience: str = "dev") -> Iterator[Usage]:
        """Reserve the cost of max_seconds of audio, then bill the seconds actually streamed (set usage.audio_seconds)."""
        provider, model = "meta_api", VOICE_MODEL
        self.check_kill_switch()
        self._check_model(model, provider)
        est = self.pricing.cost(provider, model, Usage(audio_seconds=max_seconds))
        res = self.guard.check_and_reserve(provider, model, est, run_id, audience)
        usage = Usage()
        try:
            with self.limiter.voice_slot():
                yield usage
        except BaseException:
            res.commit(self.pricing.cost(provider, model, usage), usage, note="session ended with an error")
            raise
        res.commit(self.pricing.cost(provider, model, usage), usage)

    async def voice_transcribe(
        self,
        *,
        audio: AsyncIterator[bytes],
        on_event: Callable[[dict[str, Any]], Awaitable[None]],
        run_id: str,
        max_seconds: float = 60.0,
        audience: str = "dev",
        keywords: list[str] | None = None,
    ) -> Usage:
        """Relay raw 24 kHz s16le mono PCM to the realtime endpoint and hand every JSON event to on_event.

        Billed by whole seconds streamed. The audio source should already be real time (a microphone,
        silence included); this also never lets the stream run more than 5 s ahead of the wall clock.
        """
        self.check_kill_switch()
        handshake: dict[str, Any] = {
            "authorization": {"accessToken": f"Bearer {self._voice_key()}"},  # headers are ignored by this endpoint
            "model": VOICE_MODEL, "audioEncoding": "PCM_24KHZ", "mode": "ENDPOINTING", "partialMode": "CUMULATIVE",
        }
        if keywords:
            handshake["keywords"] = keywords
        with self.voice_session(run_id=run_id, max_seconds=max_seconds, audience=audience) as usage:
            async with self._voice_connect(VOICE_URL, max_size=2**20, open_timeout=10) as ws:
                await ws.send(json.dumps(handshake))
                pump = asyncio.create_task(self._pump_audio(ws, audio, usage, max_seconds))
                try:
                    async for raw in ws:
                        if isinstance(raw, str):
                            await on_event(json.loads(raw))
                except ConnectionClosed as e:
                    if e.rcvd is not None and e.rcvd.code == 1013:
                        raise CostGuardError("Voice: the provider's stream limit is reached; try again in a moment.") from e
                    raise ProviderCallFailed(f"Voice stream closed unexpectedly ({e}).") from e
                finally:
                    pump.cancel()
            log.info("paid-call provider=meta_api model=%s run=%s audio_seconds=%.1f", VOICE_MODEL, run_id, usage.audio_seconds)
        return usage

    async def _pump_audio(self, ws: Any, audio: AsyncIterator[bytes], usage: Usage, max_seconds: float) -> None:
        start = time.monotonic()
        try:
            async for chunk in audio:
                self.check_kill_switch()
                if usage.audio_seconds >= max_seconds:
                    break
                ahead = usage.audio_seconds - (time.monotonic() - start)
                if ahead > VOICE_MAX_AHEAD_S:
                    await asyncio.sleep(ahead - VOICE_MAX_AHEAD_S)
                await ws.send(chunk)
                usage.audio_seconds += len(chunk) / VOICE_BYTES_PER_SECOND
        finally:
            # Ending the stream lets the service flush the last turn and close with code 1000.
            await ws.send(json.dumps({"type": "endStream"}))

    def _voice_key(self) -> str:
        if self.settings.mock_models:
            return "mock"
        key = os.environ.get(self.allowlist.api_key_env("meta_api"), "").strip()
        if not key:
            raise CostGuardError("MODEL_API_KEY is not set. Add it to .env (never commit it).")
        return key


# ---------- response parsing ----------

def _get(obj: Any, *path: str) -> Any:
    for name in path:
        if obj is None:
            return None
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return obj


def _dump(resp: Any) -> dict[str, Any]:
    try:
        return resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {}
    except (TypeError, ValueError) as e:  # raw payload is diagnostic only; losing it must not lose the result
        log.warning("Could not serialize response for cache: %s", e)
        return {}


def _int(value: Any) -> int:
    return int(value or 0)


def _parse_responses(resp: Any) -> _Parsed:
    u = _get(resp, "usage")
    usage = None if u is None else Usage(
        input_tokens=_int(_get(u, "input_tokens")),
        cached_tokens=_int(_get(u, "input_tokens_details", "cached_tokens")),
        output_tokens=_int(_get(u, "output_tokens")),
        reasoning_tokens=_int(_get(u, "output_tokens_details", "reasoning_tokens")),
    )
    summaries = []
    for item in _get(resp, "output") or []:
        if _get(item, "type") == "reasoning":
            summaries += [_get(s, "text") or "" for s in _get(item, "summary") or []]
    reason = _get(resp, "incomplete_details", "reason")
    text = _get(resp, "output_text") or ""
    return _Parsed(text=text, usage=usage, reasoning="\n".join(summaries) or None,
                   finish_reason=reason or _get(resp, "status"),
                   truncated=reason == "max_output_tokens", raw=_dump(resp))


def _parse_chat(resp: Any) -> _Parsed:
    u = _get(resp, "usage")
    usage = None if u is None else Usage(
        input_tokens=_int(_get(u, "prompt_tokens")),
        cached_tokens=_int(_get(u, "prompt_tokens_details", "cached_tokens")),
        output_tokens=_int(_get(u, "completion_tokens")),
        reasoning_tokens=_int(_get(u, "completion_tokens_details", "reasoning_tokens")),
    )
    choice = (_get(resp, "choices") or [None])[0]
    message = _get(choice, "message")
    finish = _get(choice, "finish_reason")
    return _Parsed(text=_get(message, "content") or "", usage=usage, reasoning=_get(message, "reasoning_content"),
                   finish_reason=finish, truncated=finish == "length", raw=_dump(resp))


def _parse_sam(resp: Any) -> _Parsed:
    frames = _get(resp, "usage", "video_frames_processed")
    usage = None if frames is None else Usage(video_frames=_int(frames))
    return _Parsed(text=_get(resp, "output_text") or "", usage=usage, finish_reason=_get(resp, "status"), raw=_dump(resp))


def _estimate_input_tokens(obj: Any) -> int:
    """Pessimistic token estimate: images at a flat rate (their base64 is not counted as text)."""
    images = 0
    chars = 0

    def walk(o: Any) -> None:
        nonlocal images, chars
        if isinstance(o, dict):
            if o.get("type") in ("input_image", "image_url"):
                images += 1
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list | tuple):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            chars += len(o)
        elif o is not None:
            chars += len(json.dumps(o, default=str))

    walk(obj)
    return math.ceil(chars / CHARS_PER_TOKEN * 1.2) + images * IMAGE_TOKEN_ESTIMATE + 16


def _retry_after(e: openai.APIStatusError) -> float | None:
    try:
        value = e.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (AttributeError, ValueError):
        return None


def _error_message(e: openai.APIStatusError) -> str:
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body) if isinstance(body, dict) else {}
    return str(err.get("message") if isinstance(err, dict) and err.get("message") else e.message)[:300]
