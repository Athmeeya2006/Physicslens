"""The real-world cases table (scripts/build_cases.py): every row is complete, its status is a rung of the ladder,
each row's number comes from its committed result, and the README table is the one the builder writes."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_cases", ROOT / "scripts" / "build_cases.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

FIELDS = ("id", "title", "problem", "who", "gives", "status", "status_rule", "cant", "link", "models")


def test_every_row_is_complete_and_its_status_is_a_rung():
    rows = B.cases()
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert all(k in r for k in FIELDS), r["id"]
        assert r["status"] in B.LADDER and r["checked"] in B.LADDER
        assert r["link"] is None or r["link"].startswith("/")


def test_new_cases_read_their_numbers_from_their_results():
    rows = {r["id"]: r for r in B.cases()}
    drone = json.loads((ROOT / "data/drone/result.json").read_text())
    assert f"{drone['heldout_total_rms_n']['physicslens']:.2f} N" in rows["drone"]["number"]
    rocket = json.loads((ROOT / "data/rocket/result.json").read_text())["summary"]
    assert f"{rocket['bucket_edge_error_median_s']:.1f} s" in rows["rocket"]["number"]
    aloha = json.loads((ROOT / "data/robot/aloha/result.json").read_text())
    assert f"{aloha['heldout']['identified']['arm_total']['rms_deg']:.2f}°" in rows["aloha"]["number"]
    assert rows["aloha"]["status"] in ("proven", "improves a public tool")


def test_readme_table_is_the_one_the_builder_writes():
    readme = (ROOT / "README.md").read_text()
    assert B.table(B.cases()) in readme
