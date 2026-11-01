"""The command line: the same engine as the site, with exit codes a script can use."""
import json
from pathlib import Path

from physicslens.cli import main

from tests.test_lens import board_video

ROOT = Path(__file__).resolve().parents[1]


def test_sky_list_and_simulate(capsys):
    assert main(["sky", "list"]) == 0
    assert "apophis_sun_only" in capsys.readouterr().out
    assert main(["sky", "simulate", "apophis_sun_only", "--years", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["target"] and out["elements_end"]


def test_sky_refuses_a_missing_scenario_with_a_reason(capsys):
    assert main(["sky", "simulate"]) == 2
    assert main(["sky", "simulate", "no_such_scenario"]) == 2
    assert capsys.readouterr().err


def test_lens_calibrate_writes_json(tmp_path, capsys):
    out = tmp_path / "lens.json"
    assert main(["lens", "calibrate", str(board_video(tmp_path / "b.mp4")), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["views"] >= 8
    assert "focal" in capsys.readouterr().out


def test_analyze_checks_its_inputs_before_spending(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "missing.mp4")]) == 2
    bad = tmp_path / "lens.json"
    bad.write_text('{"width": 3}')
    assert main(["analyze", str(ROOT / "samples" / "smoke_ball.mp4"), "--lens", str(bad), "--mock"]) == 2
    assert "Lens calibration" in capsys.readouterr().err


def test_analyze_mock_runs_the_plumbing_and_stops_plainly(tmp_path, capsys):
    out = tmp_path / "out"
    code = main(["analyze", str(ROOT / "samples" / "smoke_ball.mp4"), "--mock", "--out", str(out), "--export", "json"])
    # Canned answers are not valid scene JSON, so the run stops at scene understanding: reported, not a traceback.
    assert code == 1
    assert "stopped: The reasoning model returned invalid scene output" in capsys.readouterr().err


def test_analyze_writes_every_export_for_a_verified_result(tmp_path, capsys, monkeypatch):
    from physicslens import pipeline
    stored = json.loads((ROOT / "gallery" / "gallery-pendulum" / "result.json").read_text())
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **k: stored)  # the analysis itself is tested elsewhere
    out = tmp_path / "out"
    assert main(["analyze", str(ROOT / "samples" / "smoke_ball.mp4"), "--mock", "--out", str(out),
                 "--export", "json,py,mjcf,urdf,sdf"]) == 0
    for ext in ("json", "py", "xml", "urdf", "sdf"):
        assert (out / f"physicslens.{ext}").stat().st_size > 100
    text = capsys.readouterr().out
    assert text.startswith("accepted: Pendulum with linear damping") and "g = 9.90947" in text


def test_sky_discover_prints_the_law(capsys):
    assert main(["sky", "discover", "apophis_sun_only", "--noise-px", "2"]) == 0
    found = json.loads(capsys.readouterr().out)
    assert found["status"] == "accepted" and found["source"] == "simulation, no pixels"


def test_lens_calibrate_reports_a_failure(tmp_path, capsys):
    import cv2
    import numpy as np
    blank = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(blank), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(10):
        w.write(np.full((240, 320, 3), 128, np.uint8))
    w.release()
    assert main(["lens", "calibrate", str(blank), "--out", str(tmp_path / "l.json")]) == 1
    assert "Calibration failed" in capsys.readouterr().err and not (tmp_path / "l.json").exists()
