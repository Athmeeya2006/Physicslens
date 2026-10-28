"""Structured (Pydantic-validated) calls to the reasoning model.

The submitted build reasons on Muse Glimmer (open weights, self-hosted). Muse Spark 1.3 on the Meta API is a
development convenience only: demo and production refuse to reason on anything but the open-weight model.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..costguard import CostGuardClient, CostGuardError
from ..costguard.errors import CircuitOpen, ProviderCallFailed

log = logging.getLogger("physicslens.agent")
T = TypeVar("T", bound=BaseModel)

# JSON-schema keywords kept for constrained decoding; range/length limits are enforced by Pydantic instead,
# since support for them in strict mode isn't confirmed.
_SCHEMA_KEYS = {"type", "properties", "required", "enum", "items", "anyOf", "additionalProperties", "description", "const"}


class LLMError(RuntimeError):
    pass


class OpenWeightUnavailable(CostGuardError):
    """The submitted build must reason on the open-weight model; it never silently switches to the Meta API.

    Subclasses CostGuardError so judges and public visitors see the polite busy message.
    """


def fallback_allowed(client: CostGuardClient) -> bool:
    return client.settings.app_env == "dev"


def glimmer_provider() -> str | None:
    """"glimmer" (our own llama-server) or "together" (the same open weights, hosted), else None.

    Chosen explicitly by the machine serving the app, never inferred, so a run's provider is never a surprise.
    """
    p = os.environ.get("REASONING_PROVIDER", "").strip().lower()
    return p if p in ("glimmer", "together") else None


def glimmer_selected() -> bool:
    return glimmer_provider() is not None


@dataclass(frozen=True)
class LLMCall:
    provider: str
    model: str
    reasoning: str | None
    cost_usd: float
    cached: bool


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema with $refs inlined and only portable keywords kept."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return clean(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            out = {k: clean(v) for k, v in node.items() if k in _SCHEMA_KEYS and k != "properties"}
            if "properties" in node:
                # Keys under "properties" are field names, not schema keywords: keep them all.
                out["properties"] = {name: clean(sub) for name, sub in node["properties"].items()}
            return out
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    schema = clean(raw)
    schema["additionalProperties"] = False
    return schema


def image_b64(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()



# Some models echo parts of the JSON schema next to a correct answer; none of our schemas has fields with these
# names, so they are dropped rather than costing a full retry.
_SCHEMA_ECHO = {"type", "required", "additionalProperties", "properties", "$schema", "title"}


def _validate(schema_model: type[T], raw_text: str) -> T:
    text = _strip_fences(raw_text)
    try:
        return schema_model.model_validate_json(text)
    except ValidationError as first:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise first from None
        if not isinstance(data, dict) or not (_SCHEMA_ECHO & data.keys()) or (_SCHEMA_ECHO & schema_model.model_fields.keys()):
            raise
        return schema_model.model_validate({k: v for k, v in data.items() if k not in _SCHEMA_ECHO})


def ask_json(
    client: CostGuardClient,
    *,
    task: str,
    system: str,
    text: str,
    schema_model: type[T],
    run_id: str,
    images: list[bytes] | None = None,
    audience: str = "dev",
    meta_effort: str = "low",
    max_output_tokens: int = 2000,
    run_cap_usd: float | None = None,
) -> tuple[T, LLMCall]:
    """System prompt and schema come first and stay fixed per task so provider prompt caching applies."""
    images = images or []
    error_note = ""
    for attempt in range(2):  # one retry on malformed output, then fail clearly
        prompt = text + error_note
        call = _call(client, task=task, system=system, text=prompt, schema_model=schema_model, run_id=run_id,
                     images=images, audience=audience, meta_effort=meta_effort, max_output_tokens=max_output_tokens,
                     run_cap_usd=run_cap_usd)
        raw_text, info = call
        try:
            return _validate(schema_model, raw_text), info
        except ValidationError as e:
            log.warning("%s: invalid JSON from %s (attempt %d): %s", task, info.model, attempt + 1, e.errors()[:3])
            error_note = f"\n\nYour previous answer was invalid: {e.errors()[:3]}. Reply with JSON matching the schema exactly."
    raise LLMError(f"The reasoning model returned invalid {task} output twice; stopping instead of guessing.")


def _call(client: CostGuardClient, *, task: str, system: str, text: str, schema_model: type[BaseModel], run_id: str,
          images: list[bytes], audience: str, meta_effort: str, max_output_tokens: int,
          run_cap_usd: float | None = None) -> tuple[str, LLMCall]:
    schema = strict_schema(schema_model)
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content += [{"type": "image_url", "image_url": {"url": image_b64(j)}} for j in images]
    sys_prompt = f"{system}\n\nReply with only a JSON object matching this JSON schema:\n{json.dumps(schema)}"
    if glimmer_selected():
        try:
            # Glimmer always thinks before answering, so it gets room for reasoning on top of the answer.
            r = client.glimmer_chat(messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}],
                                    run_id=run_id, audience=audience, max_tokens=max(client.glimmer_max_tokens_default,
                                                                                     max_output_tokens * 4),
                                    run_cap_usd=run_cap_usd, provider=glimmer_provider() or "glimmer")
        except (ProviderCallFailed, CircuitOpen) as e:
            raise OpenWeightUnavailable(f"Muse Glimmer failed for {task}: {e}") from e
        return r.text, LLMCall(glimmer_provider() or "glimmer", r.model, r.reasoning, r.cost_usd, r.cached)
    if not fallback_allowed(client):
        raise OpenWeightUnavailable("The open-weight model is not configured (REASONING_PROVIDER=glimmer and GLIMMER_BASE_URL).")
    meta_content = [{"type": "input_text", "text": text}] + [{"type": "input_image", "image_url": image_b64(j)} for j in images]
    r = client.meta_text(
        input=[{"role": "user", "content": meta_content}], instructions=system, schema=schema, schema_name=task,
        effort=meta_effort, max_output_tokens=max_output_tokens, prompt_cache_key=f"physicslens-{task}-v1",
        run_id=run_id, audience=audience, run_cap_usd=run_cap_usd,
    )
    return r.text, LLMCall("meta_api", r.model, r.reasoning, r.cost_usd, r.cached)
