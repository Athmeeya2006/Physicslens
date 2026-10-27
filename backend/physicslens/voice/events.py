"""Interpreting Muse Voice Transcribe realtime events."""

from __future__ import annotations

from typing import Any

# TODO(verify): the event name `speechComplete` and the `turnId` key are documented; the key holding the text
# is not. These candidates are checked in order and the first real stream is saved to tests/fixtures to pin it.
_KIND_KEYS = ("type", "event", "eventType")
_TEXT_KEYS = ("text", "transcript", "utterance")


def _kind(event: dict[str, Any]) -> str | None:
    return next((event[k] for k in _KIND_KEYS if isinstance(event.get(k), str)), None)


def _text(event: dict[str, Any]) -> str | None:
    return next((event[k] for k in _TEXT_KEYS if isinstance(event.get(k), str)), None)


def final_turn(event: dict[str, Any]) -> tuple[str, str] | None:
    """(turnId, text) for a finished utterance; None for partials and other events."""
    if _kind(event) != "speechComplete":
        return None
    text = _text(event)
    return (str(event.get("turnId", "")), text.strip()) if text else None


def partial_text(event: dict[str, Any]) -> str | None:
    """Running transcript of the current utterance (partials carry no turnId)."""
    if "turnId" in event or _kind(event) == "speechComplete":
        return None
    return _text(event)
