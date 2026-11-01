"""Share images and share pages for a result: the right size, the right words, never a broken picture."""
import json
from html.parser import HTMLParser
from pathlib import Path

from physicslens.export.og import H, W, headline, render, share_page
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PENDULUM = ROOT / "gallery" / "gallery-pendulum"


def _result() -> dict:
    return json.loads((PENDULUM / "result.json").read_text())


def test_share_image_is_link_preview_size(tmp_path):
    out = render(_result(), PENDULUM / "poster.jpg", tmp_path / "og.jpg")
    assert Image.open(out).size == (W, H) == (1200, 630)


def test_unreadable_law_and_missing_poster_still_give_an_image(tmp_path):
    bad = {"status": "accepted", "equation": {"title": "Odd law", "latex": r"\frac{"},
           "parameters": [{"name": "k", "value": 1.5, "std": 0.1, "unit": "1/s"}]}
    out = render(bad, tmp_path / "missing.jpg", tmp_path / "og.jpg")
    assert Image.open(out).size == (W, H)


def test_headline_reports_the_first_parameter_or_the_honest_outcome():
    assert headline(_result()).startswith("g = 9.9") and headline(_result()).endswith("m/s²")
    assert headline({"status": "no_fit", "parameters": []}) == "No clean equation: the honest answer"


def test_share_page_carries_preview_metadata_and_forwards_to_the_result():
    page = share_page("gallery-pendulum", {**_result(), "reason": 'quotes " and <tags>'}, "https://example.org")
    metas: dict[str, str] = {}

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag == "meta" and "content" in a:
                metas[a.get("property") or a.get("name") or a.get("http-equiv")] = a["content"]

    P().feed(page)
    assert metas["og:image"] == "https://example.org/gallery/gallery-pendulum/og.jpg"
    assert metas["refresh"].endswith("url=/?run=gallery-pendulum")
    assert metas["twitter:card"] == "summary_large_image"
    assert "<tags>" not in page and "&lt;tags&gt;" in page


def test_share_image_endpoint_draws_once_and_refuses_unfinished_runs():
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    from .test_exports import state_for
    from .test_regime import pendulum
    from .test_whatif import finished_run
    run_id = finished_run(app_module, state_for(pendulum([(6, 0.6, 0.05)]), "pendulum", "pendulum_linear_damping"))
    c = TestClient(app_module.app)
    r = c.get(f"/api/runs/{run_id}/og.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert c.get("/api/runs/no-such-run/og.jpg").status_code == 404
