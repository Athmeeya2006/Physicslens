"""Interpreting Muse Voice Transcribe realtime events.

Verified on a real stream (Oct 27, 2026; recorded in tests/fixtures/voice_events.json). Events seen:
{"sessionId": ...}, {"type": "audioProgress"}, {"type": "speechStart", "turnId"},
{"type": "transcript", "transcript", "final": false}  (running partial, no turnId),
{"type": "speechEnd", "turnId"}, {"type": "speechComplete", "turnId", "transcript"}  (the finished utterance),
and a closing {"type": "transcript", "transcript": "", "final": true} when the stream ends.
"""

from __future__ import annotations

from typing import Any


def final_turn(event: dict[str, Any]) -> tuple[str, str] | None:
    """(turnId, text) for a finished utterance; None for everything else."""
    if event.get("type") != "speechComplete":
        return None
    text = str(event.get("transcript") or "").strip()
    return (str(event.get("turnId", "")), text) if text else None


def partial_text(event: dict[str, Any]) -> str | None:
    """Running transcript of the utterance in progress."""
    if event.get("type") != "transcript" or event.get("final"):
        return None
    return str(event.get("transcript") or "") or None
