"""What-ifs: parameter changes re-simulated by the math; voice relayed through the cost guard (mocked socket)."""

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest
from physicslens.costguard import KillSwitchActive
from physicslens.costguard.mock import responses_result
from physicslens.physics.coords import to_coordinates
from physicslens.physics.fit import fit_model
from physicslens.physics.models import MODELS
from physicslens.physics.whatif import Change, WhatIfError, simulate_whatif
from physicslens.voice.events import final_turn, partial_text

from .test_regime import FPS, pendulum

PPM = 500.0
ORIGIN = (320.0, 100.0)


@pytest.fixture(scope="module")
def fit_state():
    traj = pendulum([(8, 0.6, 0.05)])
    c = to_coordinates(traj, "pendulum")
    f = fit_model(MODELS["pendulum_linear_damping"], c)
    return {"model_id": f.model_id, "params": f.params, "y0": [float(v) for v in f.y0],
            "ctx": {k: float(v) for k, v in c.ctx.items()}, "fps": FPS, "origin_px": list(ORIGIN),
            "px_per_m": PPM, "first_frame": 0}


def zero_crossings(points) -> list[float]:
    x = np.array([p[1] for p in points])
    x = x - np.median(x)
    idx = np.nonzero(np.diff(np.sign(x)) != 0)[0]
    return [i / FPS for i in idx]


def test_longer_string_slows_the_swing(fit_state):
    base = simulate_whatif(fit_state, [], 8)
    longer = simulate_whatif(fit_state, [Change("L", "multiply", 2.0)], 8)
    assert longer["length_m"] == pytest.approx(2 * fit_state["ctx"]["length_m"])
    assert longer["period_s"] / base["period_s"] == pytest.approx(math.sqrt(2), rel=1e-6)
    # Measured on the simulated ghosts themselves, not just the formula.
    half_base = np.diff(zero_crossings(base["points"])).mean()
    half_long = np.diff(zero_crossings(longer["points"])).mean()
    assert half_long / half_base == pytest.approx(math.sqrt(2), rel=0.05)
    # Both start from the measured state, so the first ghost point is the same.
    assert base["points"][0][0] == longer["points"][0][0] == 0


def test_set_gravity_like_mars(fit_state):
    mars = simulate_whatif(fit_state, [Change("g", "set", 3.71)], 4)
    assert mars["params"]["g"] == 3.71 and mars["period_s"] > simulate_whatif(fit_state, [], 4)["period_s"]


@pytest.mark.parametrize("change", [Change("mass", "set", 2.0), Change("L", "set", -1.0), Change("g", "set", 500.0)])
def test_rejects_unknown_or_unphysical_changes(fit_state, change):
    with pytest.raises(WhatIfError):
        simulate_whatif(fit_state, [change], 2)


def test_voice_event_parsing_on_a_real_stream():
    """Events recorded from a real Muse Voice Transcribe session (the word "pendulum")."""
    events = [e["event"] for e in json.loads((Path(__file__).parent / "fixtures" / "voice_events.json").read_text())]
    finals = [final_turn(e) for e in events if final_turn(e)]
    partials = [partial_text(e) for e in events if partial_text(e)]
    assert finals == [("0", "Pendulum")]
    assert partials == ["Pendulum"]  # the closing empty final:true event is not a partial


def pcm_chunks(seconds: float, chunk_s: float = 0.1):
    n = round(seconds / chunk_s)

    async def gen():
        for _ in range(n):
            yield b"\x00\x00" * int(24_000 * chunk_s)
    return gen()


def test_voice_stream_bills_whole_seconds_and_follows_protocol(harness):
    events: list[dict] = []

    async def on_event(e):
        events.append(e)

    sockets = []

    def connect(url, **kw):
        from physicslens.costguard.mock import mock_voice_connect
        sock = mock_voice_connect(url, **kw)
        sockets.append((url, sock))
        return sock

    harness.client._voice_connect = connect
    usage = asyncio.run(harness.client.voice_transcribe(audio=pcm_chunks(1.2), on_event=on_event, run_id="v"))
    url, sock = sockets[0]
    handshake = json.loads(sock.sent[0])
    assert url == "wss://api.meta.ai/v1/asr/realtime"
    assert handshake["model"] == "muse-voice-transcribe-1.0" and handshake["mode"] == "ENDPOINTING"
    assert handshake["audioEncoding"] == "PCM_24KHZ" and handshake["authorization"]["accessToken"].startswith("Bearer ")
    assert json.loads(sock.sent[-1]) == {"type": "endStream"}
    assert usage.audio_seconds == pytest.approx(1.2)
    assert final_turn(events[0])[1] == "what if the string were twice as long"
    row = harness.client.ledger.rows()[-1]
    assert row["model"] == "muse-voice-transcribe-1.0" and row["est_cost_usd"] == pytest.approx(2 * 0.18 / 3600)


def test_kill_switch_blocks_voice(harness, settings):
    settings.kill_file.touch()

    async def on_event(e):
        pass

    with pytest.raises(KillSwitchActive):
        asyncio.run(harness.client.voice_transcribe(audio=pcm_chunks(0.5), on_event=on_event, run_id="v"))


def finished_run(app_module, fit_state) -> str:
    run_dir = app_module.runs.new_run_dir()
    result = {"status": "accepted", "clip": {"frames": 120, "fps": FPS}, "fit_state": fit_state}
    (run_dir / "result.json").write_text(json.dumps(result))
    return run_dir.name


PLAN = {"understood": True, "changes": [{"parameter": "L", "operation": "multiply", "value": 2.0}],
        "description": "String twice as long", "reply": "Simulating the same swing with a string twice as long."}


def test_typed_whatif_endpoint(fit_state):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    run_id = finished_run(app_module, fit_state)
    app_module.client._sdk("meta_api", 60).script.append(responses_result(text=json.dumps(PLAN)))
    r = TestClient(app_module.app).post(f"/api/runs/{run_id}/whatif", json={"question": "what if the string were twice as long"})
    body = r.json()
    assert r.status_code == 200 and body["understood"] and body["description"] == "String twice as long"
    assert body["length_m"] == pytest.approx(2 * fit_state["ctx"]["length_m"]) and len(body["points"]) == 120


def test_voice_socket_end_to_end(fit_state):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    run_id = finished_run(app_module, fit_state)
    app_module.client._sdk("meta_api", 60).script.append(responses_result(text=json.dumps(PLAN)))
    with TestClient(app_module.app).websocket_connect("/ws/voice") as ws:
        ws.send_json({"run_id": run_id})
        for _ in range(5):
            ws.send_bytes(b"\x00\x00" * 2400)  # 0.1 s of silence each
        ws.send_text(json.dumps({"type": "stop"}))
        transcript = ws.receive_json()
        answer = ws.receive_json()
        done = ws.receive_json()
    assert transcript == {"type": "transcript", "text": "what if the string were twice as long", "final": True}
    assert answer["type"] == "whatif" and answer["understood"] and answer["points"]
    assert done["type"] == "voice_done" and done["audio_seconds"] == pytest.approx(0.5)


def test_uncalibrated_clip_refuses_absolute_lengths(harness, fit_state):
    from physicslens.agent.whatif import run_whatif

    set_plan = {**PLAN, "changes": [{"parameter": "L", "operation": "set", "value": 0.3}], "description": "30 cm string"}
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(set_plan)))
    result = {"clip": {"frames": 120, "fps": FPS}, "fit_state": fit_state, "calibration": {"source": "none"}}
    out = run_whatif(harness.client, "what if the string were 30 cm", result, "w1")
    assert not out["understood"] and "no scale reference" in out["reply"]


def test_uncalibrated_clip_still_scales(harness, fit_state):
    from physicslens.agent.whatif import run_whatif

    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(PLAN)))
    result = {"clip": {"frames": 120, "fps": FPS}, "fit_state": fit_state, "calibration": {"source": "none"}}
    out = run_whatif(harness.client, "what if the string were twice as long", result, "w2")
    assert out["understood"] and out["length_m"] == pytest.approx(2 * fit_state["ctx"]["length_m"])


def test_unparseable_answer_is_a_polite_no_not_a_500(harness, fit_state):
    from physicslens.agent.whatif import run_whatif

    harness.sdk("meta_api").script.extend([responses_result(text="not json"), responses_result(text="still not json")])
    out = run_whatif(harness.client, "what if?", {"clip": {"frames": 120, "fps": FPS}, "fit_state": fit_state}, "w3")
    assert not out["understood"] and "couldn't" in out["reply"]
