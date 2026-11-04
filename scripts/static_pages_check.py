"""Every page of the static build renders its stored content with no analysis server: a quick check for CI.

  python scripts/static_pages_check.py [BASE_URL]     # default http://127.0.0.1:8777 (frontend/out served as files)

Opens each page once at desktop width, waits for it to settle, and fails if a script error is thrown, an error
message is visible, or the page shows less than a page's worth of text (the satellites page once rendered only an
error). The full click-through of every control is scripts/site_audit.py. Free: it never uploads.
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

PAGES = ["index", "cases", "predictions", "timing", "fireball", "sun", "satellites", "occultation", "asteroids",
         "audit", "dataset", "health", "flight", "robot", "sky", "study", "live", "glossary", "index?run=gallery-pendulum",
         "index?run=gallery-paperbag", "index?run=gallery-apophis"]
MIN_WORDS = 120


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777").rstrip("/")
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name in PAGES:
            page = browser.new_page(viewport={"width": 1366, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            path = name.replace("?", ".html?", 1) if "?" in name else f"{name}.html"
            page.goto(f"{base}/{path}", wait_until="load")
            page.wait_for_timeout(2500)
            words = len(page.inner_text("main").split()) if page.locator("main").count() else 0
            visible_errors = [t for t in page.locator(".error:visible, [role=alert]:visible").all_inner_texts() if t.strip()]
            bad = errors or visible_errors or words < MIN_WORDS
            print(f"{'FAIL' if bad else 'ok  '} {name}: {words} words" + (f", script errors {errors}" if errors else "")
                  + (f", visible errors {visible_errors}" if visible_errors else ""), flush=True)
            if bad:
                problems.append(name)
            page.close()
        browser.close()
    if problems:
        sys.exit(f"{len(problems)} page(s) failed: {', '.join(problems)}")
    print(f"all {len(PAGES)} pages render offline")


if __name__ == "__main__":
    main()
