"""Every number the README quotes from a committed result still matches that result (the test count is checked in CI)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("readme_numbers", ROOT / "scripts" / "readme_numbers.py")
numbers = importlib.util.module_from_spec(_spec)
sys.modules["readme_numbers"] = numbers  # its dataclasses look their module up while it loads
_spec.loader.exec_module(numbers)


def test_readme_numbers_match_their_sources(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["readme_numbers.py", "--no-tests"])
    assert numbers.main() == 0, capsys.readouterr().out


def test_precision_follows_how_the_number_is_written():
    assert numbers.matches("4.1e-6", 4.0856e-6) and not numbers.matches("4.1e-6", 4.2e-6)
    assert numbers.matches("38,817", 38817.3) and numbers.matches("0.023", 0.02314)
    assert numbers.matches("2.0e-10", 1.998e-10) and not numbers.matches("47", 46.4)
    assert numbers.strict(r"fitted ([\d.]+)") == r"fitted " + numbers.NUMBER
