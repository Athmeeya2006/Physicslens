"""The planets step of the Apophis demo, checked against JPL with the committed data (no network)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("orbit_demo", ROOT / "scripts" / "orbit_demo.py")
orbit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orbit)

GM_SUN_JPL = 0.0002959122082855911  # data/orbit/sun_gm.json


def test_sun_planets_and_moon_reproduce_the_2029_flyby():
    # From JPL's 2026 state, 3.3 years: the closest approach within minutes of JPL's time and within 1,000 km of its
    # 38,011 km (JPL also models the Yarkovsky push and relativity, which we do not).
    hours, km = orbit.flyby(GM_SUN_JPL, planets=True)
    assert abs(hours) < 0.1
    assert abs(km - 38011) < 1000


def test_the_sun_alone_misses_the_pass():
    hours, km = orbit.flyby(GM_SUN_JPL, planets=False)
    assert hours > 1 and km > 80000
