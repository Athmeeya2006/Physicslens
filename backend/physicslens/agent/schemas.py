"""Pydantic models for every JSON answer the reasoning model gives."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..perception.calibration import REFERENCE_SIZES
from ..physics.custom import MAX_BOUND, FreeParam, build
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
    lookalikes: bool = Field(description="True if other objects that look like the moving object are in view (the balls "
                             "of a Newton's cradle, a row of dominoes, several identical balls): a tracker can confuse them.")

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


class Located(Strict):
    moving_object_box: Box1000 | None = Field(description="Tight box around the moving object, or null if it is not visible.")
    second_object_box: Box1000 | None = Field(description="Only when asked for a second object: its box, or null.")
    reference_box: Box1000 | None = Field(description="Only when asked for a reference object: its box, or null.")


class Hypothesis(Strict):
    model_id: ModelId  # type: ignore[valid-type]
    rationale: str = Field(description="One sentence: which numbers in the summary point to this model.")


class FreeParamIn(Strict):
    name: str = Field(description="Identifier used in the expressions, e.g. mu, k, c_d.")
    unit: str = Field(description="SI unit, e.g. 1/s, m/s^2, or empty for a pure number.")
    lower: float
    upper: float
    guess: float = Field(description="Physically sensible starting value inside [lower, upper].")

    @model_validator(mode="after")
    def _within_range(self) -> FreeParamIn:
        # A bound past the fitter's range (an upper bound of 50000 on a stiffening term) is clipped, not a reason to
        # throw away the whole proposal and pay for another; the guess is kept inside the clipped range.
        lo, hi = max(self.lower, -MAX_BOUND), min(self.upper, MAX_BOUND)
        if lo < hi:
            self.lower, self.upper, self.guess = lo, hi, min(max(self.guess, lo), hi)
        return self


class FreeModelIn(Strict):
    title: str = Field(description="Short name, e.g. 'Sliding with dry friction'.")
    acceleration: list[str] = Field(min_length=1, max_length=2, description=(
        "family line: one expression for the acceleration along the path, in s (position, m) and v (velocity, m/s). "
        "family plane: two expressions, x then y acceleration, in x, y (m, y up) and vx, vy (m/s)."))
    params: list[FreeParamIn] = Field(max_length=4)
    rationale: str = Field(description="One sentence: which numbers in the summary point to this law.")


class FitWindow(Strict):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    reason: str = Field(max_length=200)


class Hypotheses(Strict):
    family: Family
    candidates: list[Hypothesis] = Field(min_length=1, max_length=4)
    free_models: list[FreeModelIn] = Field(max_length=3, description=(
        "Only for family line or plane: equations you write yourself when no listed model fits the physics. Otherwise []."))
    fit_window: FitWindow | None = Field(default=None, description=(
        "Optional: the stretch of the clip (seconds from its start) that is free motion, when its start or end is not "
        "(a hand throwing, holding or catching it). null to fit everything. It must keep at least 70% of the clip."))

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
    action: Literal["explain", "request_experiment", "try_family", "give_up", "test_models", "inspect"]
    message: str = Field(description=(
        "Plain-language text for the user: explanation, experiment request, or why no equation fits. For test_models "
        "and inspect: one sentence on what you are checking."))
    next_family: Family | None = Field(description="Only for try_family; otherwise null.")
    reasoning_summary: str = Field(description="Two or three sentences summarizing your reasoning, shown in the UI.")
    model_ids: list[str] = Field(default_factory=list, max_length=3, description=(
        "Only for test_models: up to 3 untested listed model ids of the current family; otherwise []."))
    free_model: FreeModelIn | None = Field(default=None, description=(
        "Only for test_models in family line or plane: one law you write; otherwise null."))
    window: FitWindow | None = Field(default=None, description="Only for inspect: the stretch of the clip to look at.")
    expectation: str | None = Field(default=None, description=(
        "Only for test_models and inspect: what you expect to see, and what result would change your mind."))

    @field_validator("message", "reasoning_summary")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)

    @field_validator("expectation")
    @classmethod
    def no_dashes_opt(cls, v: str | None) -> str | None:
        return None if v is None else _plain_dashes(v)


class Challenge(Strict):
    model_id: str | None = Field(description=(
        "A listed model id from the SAME family as the accepted law, not already tested; null when you write a law."))
    free_model: FreeModelIn | None = Field(description="Only for family line or plane: a law you write; otherwise null.")
    why: str = Field(description="One sentence: what in the numbers this alternative could explain as well or better.")

    @field_validator("why")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)

    @model_validator(mode="after")
    def one_kind(self) -> Challenge:
        if (self.model_id is None) == (self.free_model is None):
            raise ValueError("give exactly one of model_id or free_model")
        return self


class SkepticReview(Strict):
    challenges: list[Challenge] = Field(max_length=3, description="Up to three alternative laws for code to fit and test.")
    concerns: list[str] = Field(max_length=3, description=(
        "Confounds a fit to this clip cannot test (camera tilt, a moving pivot, a scale reference out of the plane), "
        "each one short sentence naming what in the numbers suggests it; [] if none."))
    reasoning_summary: str = Field(description="Two sentences: how you tried to break the accepted law.")

    @field_validator("reasoning_summary")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)

    @field_validator("concerns")
    @classmethod
    def no_dashes_list(cls, v: list[str]) -> list[str]:
        return [_plain_dashes(c) for c in v]


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


class TrackCheck(Strict):
    on_target: bool = Field(description="True if the green box surrounds the named object in all three frames.")
    tracked_thing: str = Field(description="What the green box actually surrounds, in a few words.")
    reasoning_summary: str = Field(description="One sentence: what you saw in the three frames.")

    @field_validator("tracked_thing", "reasoning_summary")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)


class PlannedRecording(Strict):
    label: str = Field(description="Short name, e.g. 'Front left corner' or 'Week 1'.")
    how_to_film: str = Field(description="One or two sentences the user can follow while filming.")


ScalingLawId = Literal["constant", "power_0.5", "power_-0.5", "power_1", "power_-1", "power_2", "power", "linear"]


class PlannedSweep(Strict):
    variable: str = Field(description="The setting the user changes between recordings, e.g. 'string length'.")
    unit: str = Field(description="SI unit of that setting, e.g. 'm' or 'kg'.")
    response: str = Field(description=(
        "What is compared across recordings: 'period' (from the fitted pendulum or spring), or the name of a fitted "
        "parameter such as 'g', 'gamma', 'omega', 'zeta'."))
    values: list[float] = Field(min_length=3, max_length=6, description="The settings to record, in recording order.")
    expected_law: ScalingLawId = Field(description=(  # type: ignore[valid-type]
        "The law you expect before any data: constant, power_0.5 (y proportional to sqrt(x)), power_-0.5, power_1, "
        "power_-1, power_2, power (exponent unknown) or linear. Code tests all of them; yours is only shown."))
    why: str = Field(description="One sentence: why you expect that law.")

    @field_validator("why")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)


class StudyPlan(Strict):
    deliverable: Literal["simulation_model", "health_check", "parameter_sheet"] = Field(description=(
        "simulation_model: fitted dynamics exported to MuJoCo; health_check: compare recordings of one machine over "
        "time for wear; parameter_sheet: measured values with uncertainties."))
    quantities: list[str] = Field(min_length=1, max_length=4, description="What will be measured, in plain words.")
    recordings: list[PlannedRecording] = Field(min_length=1, max_length=4)
    done_when: str = Field(description="One sentence: what result ends the study.")
    reasoning_summary: str
    sweep: PlannedSweep | None = Field(default=None, description=(
        "Only when the goal is how one quantity depends on a setting the user can change (a scaling law): the sweep, "
        "with one planned recording per value. Otherwise null."))

    @field_validator("done_when", "reasoning_summary")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)


class StudyTurn(Strict):
    action: Literal["record_more", "finish"]
    next_recording: PlannedRecording | None = Field(description="Only for record_more; otherwise null.")
    message: str = Field(description="Two or three plain sentences to the user.")
    maintenance_note: str | None = Field(description="Only for a health_check: what the change means and what to do.")
    next_check: str | None = Field(description="Only for a health_check: when to record again, e.g. 'in two weeks'.")
    reasoning_summary: str

    @field_validator("message", "reasoning_summary")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)


class SetupMark(Strict):
    kind: Literal["place_object", "camera_position", "release_point", "push_here", "keep_clear"]
    label: str = Field(description="At most four plain words, no digits, e.g. 'release here'.")
    box: Box1000 = Field(description="Where the mark goes, on the 0-1000 grid of the image.")

    @field_validator("label")
    @classmethod
    def short_words(cls, v: str) -> str:
        # Numbers are drawn by code from the verified instruction, never by the image model or this label.
        v = " ".join(_plain_dashes(v).split())
        if not 1 <= len(v.split()) <= 4 or any(c.isdigit() for c in v):
            raise ValueError("label must be one to four words with no digits")
        return v


class SetupGuidePlan(Strict):
    instruction: str = Field(description="The requested experiment restated in one plain sentence for the user.")
    marks: list[SetupMark] = Field(min_length=1, max_length=4)

    @field_validator("instruction")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)


class VoiceNotes(Strict):
    pendulum_length_m: float | None = Field(description="String or rod length if the user said one, in meters; else null.")
    object_size_m: float | None = Field(description="Longest size of the moving object if said, in meters; else null.")
    object_hint: str | None = Field(description="What to measure, as a short noun phrase, if said; else null.")
    slow_motion: float | None = Field(description="How many times slower the clip plays (e.g. 8 for 240 fps), if said.")
    label: str | None = Field(description="A name for this recording if given, e.g. 'left front shock'; else null.")
    reply: str = Field(description="One short sentence confirming what was set, or saying nothing usable was heard.")

    @field_validator("reply")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)


ControlFamily = Literal["pendulum", "oscillator", "rotation"]
ControlGoal = Literal["reach_and_hold", "swing_up", "stop", "track_sine"]


class ControlTask(Strict):
    """A control goal for an identified system. Whether a controller meets it is computed by simulation, never asked."""

    family: ControlFamily
    goal: ControlGoal
    target: float = Field(default=0.0, ge=-10.0, le=10.0, description=(
        "reach_and_hold: the angle to hold (rad from hanging) or displacement from rest (m). track_sine: the "
        "amplitude. Ignored for swing_up (upright) and stop (at rest)."))
    tolerance: float = Field(gt=0.0, le=5.0, description=(
        "Allowed error once settled: rad or m, or rad/s or m/s for stop."))
    actuator_limit: float = Field(gt=0.0, le=1000.0, description=(
        "Largest motor command per kilogram of moving mass: N m/kg at a pivot, N/kg on a slider."))
    time_limit_s: float = Field(gt=0.0, le=30.0, description="The error must be inside the tolerance by this time.")
    frequency_hz: float = Field(default=0.5, gt=0.0, le=5.0, description="track_sine only: the sine's frequency.")


class ControllerProposal(Strict):
    code: str = Field(description="Python source defining exactly one function `control`.")
    approach: str = Field(description="One or two sentences: the control law and why it should meet the goal.")

    @field_validator("approach")
    @classmethod
    def no_dashes(cls, v: str) -> str:
        return _plain_dashes(v)
