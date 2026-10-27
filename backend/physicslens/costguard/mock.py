"""Fake provider SDK for automated tests (MOCK_MODELS=1). Never used in demo/prod: settings refuses to start."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, Self

import httpx
import openai

MOCK_TEXT = "MOCK RESPONSE"


def api_error(status: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    """A real openai exception, so retry logic is exercised exactly as in production."""
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://mock.invalid"))
    cls = {400: openai.BadRequestError, 404: openai.NotFoundError, 429: openai.RateLimitError}.get(
        status, openai.InternalServerError if status >= 500 else openai.APIStatusError
    )
    return cls(f"mock HTTP {status}", response=response, body=None)


class _Raw:
    def __init__(self, parsed: Any, headers: dict[str, str]):
        self._parsed = parsed
        self.headers = headers

    def parse(self) -> Any:
        return self._parsed


class _Stream:
    def __init__(self, final: Any):
        self._final = final

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_response(self) -> Any:
        return self._final


class MockSDK:
    """Queue exceptions or responses in `script`; when it is empty, a canned success is returned."""

    def __init__(self, provider: str):
        self.provider = provider
        self.script: deque[Any] = deque()
        self.calls: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}
        self.max_retries = 0
        self.responses = SimpleNamespace(
            with_raw_response=SimpleNamespace(create=self._responses_create),
            stream=self._responses_stream,
            input_tokens=SimpleNamespace(count=lambda **kw: SimpleNamespace(input_tokens=100)),
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self._chat_create)))
        self.files = SimpleNamespace(create=self._files_create, delete=lambda file_id: None)
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id="mock-model")])

    def _next(self, kind: str, kwargs: dict[str, Any], default: Any) -> Any:
        self.calls.append({"kind": kind, **kwargs})
        item = self.script.popleft() if self.script else default
        if isinstance(item, BaseException):
            raise item
        return item

    def _responses_create(self, **kw: Any) -> _Raw:
        return _Raw(self._next("responses", kw, responses_result()), self.headers)

    def _responses_stream(self, **kw: Any) -> _Stream:
        return _Stream(self._next("stream", kw, sam_result(frames=60)))

    def _chat_create(self, **kw: Any) -> _Raw:
        return _Raw(self._next("chat", kw, chat_result()), self.headers)

    def _files_create(self, **kw: Any) -> Any:
        self.calls.append({"kind": "files", "purpose": kw.get("purpose"), "expires_after": kw.get("expires_after")})
        return SimpleNamespace(id=f"file-mock-{len(self.calls)}")


def responses_result(text: str = MOCK_TEXT, input_tokens: int = 100, output_tokens: int = 50,
                     reasoning_tokens: int = 10, cached: int = 0, incomplete: str | None = None, usage: bool = True) -> Any:
    u = SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    ) if usage else None
    return SimpleNamespace(
        output_text=text, output=[], usage=u, status="incomplete" if incomplete else "completed",
        incomplete_details=SimpleNamespace(reason=incomplete) if incomplete else None,
    )


def chat_result(text: str = MOCK_TEXT, prompt_tokens: int = 100, completion_tokens: int = 50, finish: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, reasoning_content="mock reasoning"), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                              prompt_tokens_details=None, completion_tokens_details=None),
    )


def sam_result(frames: int | None = 60, text: str = "") -> Any:
    usage = SimpleNamespace(video_frames_processed=frames) if frames is not None else None
    return SimpleNamespace(output_text=text, usage=usage, status="completed")


def mock_sdk_factory(provider: str, base_url: str, api_key_env: str, timeout: float) -> MockSDK:
    return MockSDK(provider)


class _MockVoiceSocket:
    """Stands in for the realtime transcription socket: one final turn after the stream ends."""

    def __init__(self, text: str):
        self.text = text
        self.sent: list[Any] = []
        self._ended = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def send(self, data: Any) -> None:
        self.sent.append(data)
        if isinstance(data, str) and '"endStream"' in data:
            self._ended = True

    def __aiter__(self) -> _MockVoiceSocket:
        return self

    async def __anext__(self) -> str:
        import asyncio
        import json

        while not self._ended:
            await asyncio.sleep(0.01)
        if self.text is None:
            raise StopAsyncIteration
        text, self.text = self.text, None
        return json.dumps({"type": "speechComplete", "turnId": "mock-1", "text": text})


MOCK_VOICE_TEXT = "what if the string were twice as long"


def mock_voice_connect(url: str, **kwargs: Any) -> _MockVoiceSocket:
    return _MockVoiceSocket(MOCK_VOICE_TEXT)
