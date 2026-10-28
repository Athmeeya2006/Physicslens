"""Settings spoken while filming ("the string is 50 centimeters", "this is the left front shock")."""

from __future__ import annotations

from typing import Any

from ..costguard import CostGuardClient
from .llm import LLMError, ask_json
from .schemas import VoiceNotes

NOTES_SYSTEM = """While filming, the user says things about the recording. Turn them into settings for the analysis.
Only fill what was actually said; convert units yourself (50 centimeters is 0.5 m, 2 feet is 0.61 m). A string,
rope, rod or arm length is pendulum_length_m; the size of the moving object itself is object_size_m; what to track
is object_hint; slow motion (120 fps is 4, 240 fps is 8 when played at 30) is slow_motion; a name for the recording
(which corner, which machine, which week) is label."""


def interpret_notes(client: CostGuardClient, text: str, run_id: str, audience: str = "dev") -> dict[str, Any]:
    try:
        notes, call = ask_json(client, task="voice_notes", system=NOTES_SYSTEM, text=text[:500], schema_model=VoiceNotes,
                               run_id=run_id, audience=audience, meta_effort="minimal", max_output_tokens=600)
    except LLMError:
        return {"understood": False, "heard": text, "reply": "I couldn't turn that into a setting; try 'the string is 50 centimeters'."}
    out = notes.model_dump()
    # Physically meaningless values are dropped rather than passed on to the analysis.
    for key, lo, hi in (("pendulum_length_m", 0.005, 100), ("object_size_m", 0.005, 100), ("slow_motion", 1, 1000)):
        if out[key] is not None and not lo <= out[key] <= hi:
            out[key] = None
    out.update(understood=any(out[k] is not None for k in ("pendulum_length_m", "object_size_m", "object_hint",
                                                         "slow_motion", "label")),
               heard=text, model=call.model, provider=call.provider)
    return out
