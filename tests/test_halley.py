"""The planets step of the Halley demo, checked against JPL with the committed data (no network)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("halley_demo", ROOT / "scripts" / "halley_demo.py")
halley = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(halley)

GM_SUN_JPL = 0.0002959122082855911  # data/halley/sun_gm.json


def test_sun_and_giant_planets_reproduce_jpls_2061_return():
    # 76 years from the 1985 state: Newton's pull of the Sun and the four giants lands within days of JPL's date
    # (JPL also models the comet's outgassing, which we do not), at JPL's perihelion distance.
    days, q = halley.return_date(GM_SUN_JPL, planets=True)
    assert abs(days) < 10
    assert abs(q - 0.5928) < 0.001


def test_the_sun_alone_misses_by_months():
    days, _ = halley.return_date(GM_SUN_JPL, planets=False)
    assert days > 150
