"""The outcome card renders on result pages, above the equation, with the readable law (a real browser).

  python scripts/outcome_smoke.py [site]      site defaults to http://127.0.0.1:8777 (a static build being served)

Opens two gallery results: the pendulum (an independent truth: g) and the spring cart (none). Checks the card is
there and above the equation, its status matches the result JSON, the law with numbers is typeset, and tapping a
term shows the forecast without it. Needs Playwright and Chrome; exits non-zero on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777"
ROOT = Path(__file__).resolve().parents[1]
STATUS_WORD = {"proven": "Proven", "partial": "Verified", "not yet validated": "Not yet validated"}


def main() -> int:
    failures: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        for gid in ("gallery-pendulum", "gallery-spring"):
            expected = json.loads((ROOT / "frontend" / "public" / "gallery" / gid / "result.json").read_text())["outcome"]
            page = browser.new_page(viewport={"width": 390, "height": 844})
            errors: list[str] = []
            page.on("pageerror", lambda e, errs=errors: errs.append(str(e)))
            page.goto(f"{SITE}/?run={gid}", wait_until="networkidle")
            card = page.locator("section.outcome")
            try:
                card.wait_for(timeout=15000)
            except Exception:  # noqa: BLE001  reported below
                failures.append(f"{gid}: no outcome card")
                continue
            if STATUS_WORD[expected["status"]] not in card.inner_text():
                failures.append(f"{gid}: card status is not {expected['status']!r}")
            if expected["headline"] not in card.inner_text():
                failures.append(f"{gid}: headline differs from the result JSON")
            eq = page.locator(".equation").first
            if card.bounding_box()["y"] >= eq.bounding_box()["y"]:
                failures.append(f"{gid}: the card is not above the equation")
            if not page.locator(".readable .katex").count():
                failures.append(f"{gid}: the law with numbers is not typeset")
            term = page.locator("button.term").first
            if term.count():
                term.click()
                if not page.get_by_text("Without this term the law is").count():
                    failures.append(f"{gid}: tapping a term showed no forecast without it")
            failures += [f"{gid}: page error {e}" for e in errors]
        browser.close()
    print("OK" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
