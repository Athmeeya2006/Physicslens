"""Structured (Pydantic-validated) calls to the reasoning model, with Baseten primary and Meta fallback."""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..costguard import CostGuardClient, PriceMissing
from ..costguard.errors import CircuitOpen, ProviderCallFailed
from ..settings import BASETEN_REASONING_MODEL

log = logging.getLogger("physicslens.agent")
T = TypeVar("T", bound=BaseModel)

# JSON-schema keywords kept for constrained decoding; range/length limits are enforced by Pydantic instead,
# since support for them in strict mode isn't confirmed.
_SCHEMA_KEYS = {"type", "properties", "required", "enum", "items", "anyOf", "additionalProperties", "description", "const"}


class LLMError(RuntimeError):
    pass


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
            return {k: clean(v) for k, v in node.items() if k in _SCHEMA_KEYS}
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


def baseten_available(client: CostGuardClient) -> bool:
    if not os.environ.get("BASETEN_API_KEY", "").strip():
        return False
    try:
        client.pricing.entry("baseten", BASETEN_REASONING_MODEL)
    except PriceMissing:
        return False
    return True


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
) -> tuple[T, LLMCall]:
    """System prompt and schema come first and stay fixed per task so provider prompt caching applies."""
    images = images or []
    error_note = ""
    for attempt in range(2):  # one retry on malformed output, then fail clearly
        prompt = text + error_note
        call = _call(client, task=task, system=system, text=prompt, schema_model=schema_model, run_id=run_id,
                     images=images, audience=audience, meta_effort=meta_effort, max_output_tokens=max_output_tokens)
        raw_text, info = call
        try:
            return schema_model.model_validate_json(_strip_fences(raw_text)), info
        except ValidationError as e:
            log.warning("%s: invalid JSON from %s (attempt %d): %s", task, info.model, attempt + 1, e.errors()[:3])
            error_note = f"\n\nYour previous answer was invalid: {e.errors()[:3]}. Reply with JSON matching the schema exactly."
    raise LLMError(f"The reasoning model returned invalid {task} output twice; stopping instead of guessing.")


def _call(client: CostGuardClient, *, task: str, system: str, text: str, schema_model: type[BaseModel], run_id: str,
          images: list[bytes], audience: str, meta_effort: str, max_output_tokens: int) -> tuple[str, LLMCall]:
    schema = strict_schema(schema_model)
    if baseten_available(client):
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content += [{"type": "image_url", "image_url": {"url": image_b64(j)}} for j in images]
        sys_prompt = f"{system}\n\nReply with only a JSON object matching this JSON schema:\n{json.dumps(schema)}"
        try:
            r = client.baseten_chat(messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}],
                                    run_id=run_id, audience=audience, max_tokens=max_output_tokens * 2)
            return r.text, LLMCall("baseten", r.model, r.reasoning, r.cost_usd, r.cached)
        except (ProviderCallFailed, CircuitOpen) as e:
            log.warning("Baseten failed for %s (%s); falling back to Muse Spark on the Meta API.", task, e)
    content = [{"type": "input_text", "text": text}] + [{"type": "input_image", "image_url": image_b64(j)} for j in images]
    r = client.meta_text(
        input=[{"role": "user", "content": content}], instructions=system, schema=schema, schema_name=task,
        effort=meta_effort, max_output_tokens=max_output_tokens, prompt_cache_key=f"physicslens-{task}-v1",
        run_id=run_id, audience=audience,
    )
    return r.text, LLMCall("meta_api", r.model, r.reasoning, r.cost_usd, r.cached)
