"""The series handed to the agent for each real-world case: one value per time, in order, from the committed data."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("cases_agent", ROOT / "scripts" / "cases_agent.py")
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)


@pytest.mark.parametrize("case", sorted(ca.CASES))
def test_each_case_gives_one_value_per_time_and_its_hand_chosen_laws(case):
    # The coast-down reads the public bench recording downloaded into var/ (scripts/fetch_health_public.py fetches it); a
    # clean checkout, such as CI's, does not have it.
    if case == "coastdown" and not (ROOT / "var" / "health_public" / "spindown" / "bench_spindown_lifted.csv").is_file():
        pytest.skip("the public coast-down recording is not downloaded (var/health_public)")
    d = ca.CASES[case]()
    t, y = np.asarray(d["t"]), np.asarray(d["y"])
    assert len(t) == len(y) >= 30 and t[0] == 0
    assert np.all(np.diff(np.sort(t)) > 0)  # no two values at one time (stations are binned for the fireball)
    assert np.all(np.isfinite(y)) and d["hand"] and d["provenance"]["data_source"]


def test_wasp12b_series_holds_transits_only():
    d = ca.wasp12b()
    assert len(d["t"]) < 158  # the occultations are left out
