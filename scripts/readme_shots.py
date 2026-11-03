"""Screenshots of the site for the README, taken from a static build so the README always shows the site as it is.

  python scripts/readme_shots.py [site]      -> frontend/public/readme/*.png (light theme, 1440 x 900, then 2x scaled down)
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777").rstrip("/")
OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "readme"
# name: (page, element to scroll to the top, or None for the top of the page)
SHOTS = {
    "home": ("index.html", None),
    "robot-before-after": ("robot.html", "#rb-trace"),
    "agent-reasoning": ("index.html?run=gallery-pendulum", "#ar-title"),
    "result": ("index.html?run=gallery-pendulum", None),
    "cases": ("cases.html", None),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chrome = {"executable_path": "/usr/bin/google-chrome"} if Path("/usr/bin/google-chrome").exists() else {"channel": "chromium"}
    with sync_playwright() as p:
        browser = p.chromium.launch(**chrome, headless=True)
        for name, (page_path, anchor) in SHOTS.items():
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1, reduced_motion="reduce")
            ctx.add_init_script("try{localStorage.setItem('physicslens-theme','light')}catch(e){}")
            page = ctx.new_page()
            page.goto(f"{SITE}/{page_path}", wait_until="networkidle")
            page.wait_for_timeout(2500)
            if anchor:
                page.locator(anchor).first.scroll_into_view_if_needed()
                page.evaluate(f"() => {{ const e = document.querySelector('{anchor}'); window.scrollBy(0, e.getBoundingClientRect().top - 130); }}")
                page.wait_for_timeout(800)
            page.screenshot(path=str(OUT / f"{name}.png"))
            print(OUT / f"{name}.png", flush=True)
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
