"""Pre-registration: a prediction is written once, hashed, never edited, and scored only after its date."""
import json
from datetime import UTC, datetime

import pytest
from physicslens import prereg as P

BODY = {"title": "Next transit of X b", "question": "When will X b next transit?",
        "inputs": [{"name": "TESS sector 70", "source": "MAST", "url": "https://example.org"}],
        "prediction": {"items": [{"label": "transit 1", "value": 2461000.5, "ci95": [2461000.49, 2461000.51]}]},
        "scoring": {"rule": "hit if the observed mid-time lies inside ci95", "score_after_utc": "2027-01-01T00:00:00Z"}}


def test_register_writes_once_and_indexes_the_hash(tmp_path):
    p = P.register("timing", "x-b-1", BODY, root=tmp_path, now=datetime(2026, 11, 2, tzinfo=UTC))
    doc = json.loads(p.read_text())
    assert doc["status"] == "open" and doc["created_utc"] == "2026-11-02T00:00:00Z" and "commit" in doc["code_version"]
    rows = P.index_rows(tmp_path)
    assert rows[0]["id"] == "x-b-1" and rows[0]["sha256"] == P.sha256(p)
    with pytest.raises(P.PreregError, match="never edited"):
        P.register("timing", "x-b-1", BODY, root=tmp_path)
    assert P.verify(tmp_path) == []


def test_an_edited_prediction_is_caught(tmp_path):
    p = P.register("timing", "x-b-1", BODY, root=tmp_path)
    p.write_text(p.read_text().replace("2461000.5", "2461000.6"))
    assert P.verify(tmp_path) == ["timing/x-b-1: changed after it was registered"]


def test_scoring_waits_for_the_date_and_keeps_misses(tmp_path):
    P.register("timing", "x-b-1", BODY, root=tmp_path)
    out = tmp_path / "outcomes"
    with pytest.raises(P.PreregError, match="too early"):
        P.score("timing", "x-b-1", {"items": [], "summary": "", "hits": 0, "misses": 1}, root=tmp_path, outcomes=out,
                now=datetime(2026, 12, 1, tzinfo=UTC))
    P.score("timing", "x-b-1", {"items": [{"label": "transit 1", "hit": False}], "summary": "missed", "hits": 0, "misses": 1},
            root=tmp_path, outcomes=out, now=datetime(2027, 2, 1, tzinfo=UTC))
    s = P.summary(tmp_path, out)
    assert s["scored"] == 1 and s["misses"] == 1 and s["predictions"][0]["intact"]


@pytest.mark.parametrize("bad", ["../x", "", ".hidden", "a b"])
def test_names_cannot_escape_the_folder(tmp_path, bad):
    with pytest.raises(P.PreregError):
        P.register(bad, "id", BODY, root=tmp_path)


def test_a_prediction_without_a_scoring_date_is_refused(tmp_path):
    body = {**BODY, "scoring": {"rule": "later"}}
    with pytest.raises(P.PreregError, match="score_after_utc"):
        P.register("timing", "x", body, root=tmp_path)
