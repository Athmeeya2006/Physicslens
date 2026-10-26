"""Turns provider usage into dollars using config/prices.yaml."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .errors import PriceMissing

PER_MILLION = 1_000_000


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    images: int = 0
    video_frames: int = 0
    audio_seconds: float = 0.0
    gpu_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Pricing:
    def __init__(self, prices_cfg: dict[str, Any], sam_min_frames: int = 50):
        self._prices = prices_cfg
        self.sam_min_frames = sam_min_frames

    def entry(self, provider: str, model: str) -> dict[str, float]:
        entry = (self._prices.get(provider) or {}).get(model)
        if not entry:
            raise PriceMissing(f"No price for {provider}/{model} in config/prices.yaml; refusing the call.")
        missing = [k for k, v in entry.items() if v is None]
        if missing:
            raise PriceMissing(
                f"Price for {provider}/{model} has unset fields {missing} (TODO(verify)); refusing the call."
            )
        return entry

    def cost(self, provider: str, model: str, usage: Usage, video_requests: int = 0) -> float:
        p = self.entry(provider, model)
        total = 0.0
        if "input" in p:
            cached = min(usage.cached_tokens, usage.input_tokens)
            total += (usage.input_tokens - cached) * p["input"] / PER_MILLION
            total += cached * p.get("cached_input", p["input"]) / PER_MILLION
        if "output" in p:
            # Reasoning is normally counted inside output_tokens. If a provider ever reports it
            # separately (reasoning > output), bill both: overestimating is the safe direction.
            out = usage.output_tokens
            if usage.reasoning_tokens > usage.output_tokens:
                out += usage.reasoning_tokens
            total += out * p["output"] / PER_MILLION
        if "video_frame" in p:
            frames = usage.video_frames
            if video_requests:
                frames = max(frames, self.sam_min_frames * video_requests)
            total += frames * p["video_frame"]
        if "image" in p:
            total += usage.images * p["image"]
        if "audio_hour" in p:
            total += math.ceil(usage.audio_seconds) * p["audio_hour"] / 3600
        if "gpu_second" in p:
            total += usage.gpu_seconds * p["gpu_second"]
        return total

    def worst_case_tokens(self, provider: str, model: str, input_tokens: int, max_output_tokens: int) -> float:
        """Reservation estimate: nothing cached, every output token used."""
        return self.cost(provider, model, Usage(input_tokens=input_tokens, output_tokens=max_output_tokens))
