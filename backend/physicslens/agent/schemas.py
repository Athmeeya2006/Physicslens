"""Pydantic models for every JSON answer the reasoning model gives."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..perception.calibration import REFERENCE_SIZES
from ..physics.models import MODELS

ReferenceKind = Literal[tuple(REFERENCE_SIZES) + ("none",)]  # type: ignore[valid-type]
ModelId = Literal[tuple(MODELS)]  # type: ignore[valid-type]
Family = Literal["pendulum", "oscillator", "projectile", "double_pendulum", "rotation"]

_BANNED_PHRASE_WORDS = {"segment", "track", "find", "the", "not", "and", "or", "left", "right", "above", "below", "near"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Box1000(Strict):
    """Box on the 0-1000 grid Muse Spark uses for image coordinates."""

    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)
    x2: int = Field(ge=0, le=1000)
    y2: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def ordered(self) -> Box1000:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have x2 > x1 and y2 > y1")
        return self

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        def sx(v: int) -> int:
            return min(max(round(v * width / 1000), 0), width)

        def sy(v: int) -> int:
            return min(max(round(v * height / 1000), 0), height)

        return sx(self.x1), sy(self.y1), sx(self.x2), sy(self.y2)


class Scene(Strict):
    moving_object: str = Field(description="Short concrete noun phrase for the moving object, e.g. 'metal ball'.")
    moving_object_box: Box1000 = Field(description="Box around the moving object in the FIRST image.")
    motion_family: Literal["pendulum", "oscillator", "projectile", "double_pendulum", "rotation", "other"]
    second_object: str | None = Field(description="For a double pendulum only: the outer bob (the moving_object is the inner one). Otherwise null.")
    second_object_box: Box1000 | None = Field(description="Box around the second object in the FIRST image, or null.")
    reference_kind: ReferenceKind = Field(description="Known-size object visible in the plane of motion, or 'none'.")  # type: ignore[valid-type]
    reference_box: Box1000 | None = Field(description="Box around the reference object in the FIRST image, or null.")
    observations: str = Field(description="One or two sentences on what is happening.")

    @field_validator("moving_object")
    @classmethod
    def concrete_phrase(cls, v: str) -> str:
        words = v.lower().split()
        if not 1 <= len(words) <= 4 or set(words) & _BANNED_PHRASE_WORDS or any(c in v for c in ",;:?!"):
            raise ValueError("moving_object must be a 1-4 word noun phrase, no commands, lists or spatial words")
        return v.strip()

    @model_validator(mode="after")
    def second_consistent(self) -> Scene:
        if (self.second_object is None) != (self.second_object_box is None):
            raise ValueError("second_object and second_object_box must both be set or both be null")
        if self.motion_family == "double_pendulum" and self.second_object is None:
            raise ValueError("a double pendulum needs second_object (the outer bob) and its box")
        return self

    @model_validator(mode="after")
    def reference_consistent(self) -> Scene:
        if (self.reference_kind == "none") != (self.reference_box is None):
            raise ValueError("reference_box must be null exactly when reference_kind is 'none'")
        return self


class Hypothesis(Strict):
    model_id: ModelId  # type: ignore[valid-type]
    rationale: str = Field(description="One sentence: which numbers in the summary point to this model.")


class Hypotheses(Strict):
    family: Family
    candidates: list[Hypothesis] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def same_family(self) -> Hypotheses:
        wrong = [h.model_id for h in self.candidates if MODELS[h.model_id].family != self.family]
        if wrong:
            raise ValueError(f"candidates {wrong} are not {self.family} models")
        return self


class AgentTurn(Strict):
    action: Literal["explain", "request_experiment", "try_family", "give_up"]
    message: str = Field(description="Plain-language text for the user: explanation, experiment request, or why no equation fits.")
    next_family: Family | None = Field(description="Only for try_family; otherwise null.")
    reasoning_summary: str = Field(description="Two or three sentences summarizing your reasoning, shown in the UI.")


class ParamChange(Strict):
    parameter: str = Field(description="Exactly one of the parameter names listed in the request.")
    operation: Literal["multiply", "set", "add"]
    value: float


class WhatIfPlan(Strict):
    understood: bool = Field(description="False if the question isn't a change to this system's parameters.")
    changes: list[ParamChange] = Field(max_length=3)
    description: str = Field(description="Short label for the new scenario, e.g. 'String twice as long'.")
    reply: str = Field(description="One sentence for the user: what will be simulated, or why it can't be.")
