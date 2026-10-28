"""Settings spoken while filming become analysis options."""

import json

from physicslens.agent.notes import interpret_notes
from physicslens.costguard.mock import responses_result

NOTES = {"pendulum_length_m": 0.5, "object_size_m": None, "object_hint": "the steel ball", "slow_motion": None,
         "label": "left front shock", "reply": "Set the string to 50 cm."}


def test_spoken_settings_are_parsed_and_checked(harness):
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(NOTES)))
    out = interpret_notes(harness.client, "the string is fifty centimeters, this is the left front shock", "n1")
    assert out["understood"] and out["pendulum_length_m"] == 0.5 and out["label"] == "left front shock"

    silly = {**NOTES, "pendulum_length_m": 5000.0, "object_hint": None, "label": None}
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(silly)))
    out = interpret_notes(harness.client, "the string is five kilometers", "n2")
    assert out["pendulum_length_m"] is None and not out["understood"]

    harness.sdk("meta_api").script.extend([responses_result(text="no"), responses_result(text="still no")])
    out = interpret_notes(harness.client, "mumble", "n3")
    assert not out["understood"] and "couldn't" in out["reply"]


def test_voice_notes_socket():
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    app_module.client._sdk("meta_api", 60).script.append(responses_result(text=json.dumps(NOTES)))
    with TestClient(app_module.app).websocket_connect("/ws/voice/notes") as ws:
        ws.send_json({})
        for _ in range(5):
            ws.send_bytes(b"\x00\x00" * 2400)
        ws.send_text(json.dumps({"type": "stop"}))
        transcript, notes, done = ws.receive_json(), ws.receive_json(), ws.receive_json()
    assert transcript["final"] and notes["type"] == "notes" and notes["pendulum_length_m"] == 0.5
    assert done["type"] == "voice_done"


def test_live_length_said_mid_session(harness, tmp_path):
    from physicslens.live import LiveConfig, LiveSession
    from physicslens.perception.calibration import Calibration

    sent: list[dict] = []
    s = LiveSession(harness.client, tmp_path / "l", "l1", LiveConfig(pendulum_length_m=0.4), sent.append,
                    sam_fn=lambda *a: "", synchronous=True)
    s.calibration = Calibration(500.0, 0.0, "user")
    s.set_length(0.6)
    assert s.cfg.pendulum_length_m == 0.6 and s.calibration is None and "60 cm" in sent[-1]["message"]
    s.calibration = Calibration(800.0, 0.01, "a4_sheet")
    s.set_length(0.5)
    assert s.calibration.source == "a4_sheet"  # a measured reference outranks a spoken length
