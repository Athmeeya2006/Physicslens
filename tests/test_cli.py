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
