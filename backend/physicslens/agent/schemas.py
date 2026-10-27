"""Pydantic models for every JSON answer the reasoning model gives."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..perception.calibration import REFERENCE_SIZES
from ..physics.custom import FreeParam, build
from ..physics.expr import ExprError
from ..physics.models import MODELS

ReferenceKind = Literal[tuple(REFERENCE_SIZES) + ("none",)]  # type: ignore[valid-type]
ModelId = Literal[tuple(MODELS)]  # type: ignore[valid-type]
Family = Literal["pendulum", "oscillator", "projectile", "double_pendulum", "rotation", "line", "plane"]

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


class FreeParamIn(Strict):
    name: str = Field(description="Identifier used in the expressions, e.g. mu, k, c_d.")
    unit: str = Field(description="SI unit, e.g. 1/s, m/s^2, or empty for a pure number.")
    lower: float
    upper: float
    guess: float = Field(description="Physically sensible starting value inside [lower, upper].")


class FreeModelIn(Strict):
    title: str = Field(description="Short name, e.g. 'Sliding with dry friction'.")
    acceleration: list[str] = Field(min_length=1, max_length=2, description=(
        "family line: one expression for the acceleration along the path, in s (position, m) and v (velocity, m/s). "
        "family plane: two expressions, x then y acceleration, in x, y (m, y up) and vx, vy (m/s)."))
    params: list[FreeParamIn] = Field(max_length=4)
    rationale: str = Field(description="One sentence: which numbers in the summary point to this law.")


class Hypotheses(Strict):
    family: Family
    candidates: list[Hypothesis] = Field(min_length=1, max_length=4)
    free_models: list[FreeModelIn] = Field(max_length=3, description=(
        "Only for family line or plane: equations you write yourself when no listed model fits the physics. Otherwise []."))

    @model_validator(mode="after")
    def same_family(self) -> Hypotheses:
        wrong = [h.model_id for h in self.candidates if MODELS[h.model_id].family != self.family]
        if wrong:
            raise ValueError(f"candidates {wrong} are not {self.family} models")
        if self.free_models and self.family not in ("line", "plane"):
            raise ValueError("free_models are only allowed with family 'line' or 'plane'; use listed model ids otherwise")
        for i, f in enumerate(self.free_models):
            try:
                build(self.family, f.title, f.acceleration, [FreeParam(**p.model_dump()) for p in f.params])
            except ExprError as e:
                raise ValueError(f"free_models[{i}] ({f.title}): {e}") from e
        return self


def _plain_dashes(text: str) -> str:
    """Model prose is shown in the UI, whose copy uses no em or en dashes; turn them into commas and 'to'."""
    return text.replace(" \u2014 ", ", ").replace("\u2014", ", ").replace(" \u2013 ", " to ").replace("\u2013", " to ")


class AgentTurn(Strict):
    action: Literal["explain", "request_experiment", "try_family", "give_up"]
    message: str = Field(description="Plain-language text for the user: explanation, experiment request, or why no equation fits.")
    next_family: Family | None = Field(description="Only for try_family; otherwise null.")
    reasoning_summary: str = Field(description="Two or three sentences summarizing your reasoning, shown in the UI.")

    @field_validator("message", "reasoning_summary")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)


class ParamChange(Strict):
    parameter: str = Field(description="Exactly one of the parameter names listed in the request.")
    operation: Literal["multiply", "set", "add"]
    value: float


class WhatIfPlan(Strict):
    understood: bool = Field(description="False if the question isn't a change to this system's parameters.")
    changes: list[ParamChange] = Field(max_length=3)
    description: str = Field(description="Short label for the new scenario, e.g. 'String twice as long'.")
    reply: str = Field(description="One sentence for the user: what will be simulated, or why it can't be.")

    @field_validator("description", "reply")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)
