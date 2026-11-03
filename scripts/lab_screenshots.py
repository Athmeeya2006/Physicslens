"""Screenshots of the Solar System Lab for the README, taken from a static build (not drawn by hand).

  python scripts/lab_screenshots.py [site]      -> frontend/public/readme/lab-desktop.png, lab-phone.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777"
OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "readme"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # The installed Chrome where there is one; Playwright's full Chromium on a cluster node without it.
        chrome = {"executable_path": "/usr/bin/google-chrome"} if Path("/usr/bin/google-chrome").exists() else {"channel": "chromium"}
        browser = p.chromium.launch(**chrome, headless=True)
        for name, size in (("lab-desktop", (1366, 900)), ("lab-phone", (390, 844))):
            ctx = browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=2)
            page = ctx.new_page()
            page.goto(f"{SITE}/sky.html#halley_with_jupiter_saturn", wait_until="networkidle")
            page.wait_for_timeout(1500)
            page.get_by_label("Jupiter", exact=True).first.uncheck()  # the changed run, dashed over the original
            page.wait_for_timeout(9000)  # let the orbits play for a while
            page.screenshot(path=str(OUT / f"{name}.png"))
            print(OUT / f"{name}.png")
        browser.close()


if __name__ == "__main__":
    main()
