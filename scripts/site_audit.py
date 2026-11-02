"""Click through the whole site in a real browser and report anything broken.

  python3.12 scripts/site_audit.py [BASE_URL] [--pages index,study,...] [--html]

BASE_URL defaults to http://127.0.0.1:8777 (the static build served by `python3 -m http.server` from frontend/out;
pass --html there, since that server does not map /page to page.html). Every page is opened at phone and desktop
width in the dark and the light theme; every button, tab, toggle, select, disclosure and in-page control is clicked
in turn, and after each click the audit checks: no uncaught page error, no failed same-site request, no horizontal
overflow, and that the page still shows its content (a click that blanks or blacks out the page is caught by the
visible text and by how much of the screen is near-black). Needs Playwright and Chrome. Free: it never uploads.
"""

from __future__ import annotations

import json
import sys
from urllib.parse import urlparse

from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright

PAGES = ["index", "study", "live", "health", "audit", "sky", "cases", "dataset", "timing", "fireball", "sun",
         "satellites", "occultation", "asteroids", "index?run=gallery-pendulum", "index?run=gallery-paperbag",
         "index?run=gallery-spring", "index?run=gallery-apophis"]
VIEWPORTS = {"phone": {"width": 390, "height": 844}, "desktop": {"width": 1366, "height": 900}}
CLICKABLE = ("button:visible, [role=tab]:visible, summary:visible, input[type=checkbox]:visible, "
             "input[type=radio]:visible, select:visible, [role=button]:visible")
# Controls that start a camera, a microphone, an upload or a paid analysis: their own flows are checked elsewhere.
SKIP_WORDS = ("start camera", "start live", "ask by voice", "analyze", "upload", "choose file", "record", "start study",
              "run study", "simulate", "delete", "sign", "allow")


def visible_text(page) -> int:
    return page.evaluate("() => (document.querySelector('main') || document.body).innerText.trim().length")


def dark_fraction(page) -> float:
    """Share of the viewport that is nearly black (a blacked-out page is close to 1; the dark theme is not)."""
    shot = page.screenshot()
    from io import BytesIO

    from PIL import Image
    im = Image.open(BytesIO(shot)).convert("L").resize((96, 96))
    px = list(im.get_flattened_data()) if hasattr(im, "get_flattened_data") else list(im.getdata())
    return sum(1 for v in px if v < 8) / len(px)


def audit_page(browser, base: str, path: str, vp: str, theme: str, html: bool) -> list[str]:
    problems: list[str] = []
    ctx = browser.new_context(viewport=VIEWPORTS[vp], color_scheme=theme)
    page = ctx.new_page()
    page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
    host = urlparse(base).netloc

    def on_response(r):
        if r.status >= 400 and urlparse(r.url).netloc == host and not r.url.endswith("/backend.json"):
            problems.append(f"HTTP {r.status} {urlparse(r.url).path}")
    page.on("response", on_response)
    name, _, query = path.partition("?")
    url = f"{base}/{name}{'.html' if html else ''}" if name != "index" else f"{base}/{'index.html' if html else ''}"
    url += f"?{query}" if query else ""
    try:
        page.goto(url, wait_until="networkidle", timeout=45000)
    except PWError as e:
        return [f"could not load: {e}"]
    page.wait_for_timeout(800)
    base_text = visible_text(page)
    if base_text < 80:
        problems.append(f"almost no text on load ({base_text} characters)")
    if page.evaluate("document.documentElement.scrollWidth") > VIEWPORTS[vp]["width"] + 1:
        problems.append(f"horizontal overflow on load ({page.evaluate('document.documentElement.scrollWidth')} px)")
    count = page.locator(CLICKABLE).count()
    clicked = 0
    for i in range(min(count, 60)):
        loc = page.locator(CLICKABLE).nth(i)
        try:
            label = (loc.inner_text(timeout=1000) or loc.get_attribute("aria-label") or loc.get_attribute("title") or "").strip()[:40]
        except PWError:
            continue
        if any(w in label.lower() for w in SKIP_WORDS):
            continue
        before = page.url
        try:
            tag = loc.evaluate("e => e.tagName.toLowerCase()")
            if tag == "select":
                opts = loc.locator("option")
                if opts.count() > 1:
                    loc.select_option(index=opts.count() - 1, timeout=2000)
            else:
                loc.click(timeout=2000)
        except PWError:
            continue
        clicked += 1
        page.wait_for_timeout(500)
        text = visible_text(page)
        if text < max(40, base_text * 0.2):
            problems.append(f"after clicking '{label}': the page lost its content ({text} characters left)")
        if dark_fraction(page) > 0.97:
            problems.append(f"after clicking '{label}': the screen went black")
        if page.evaluate("document.documentElement.scrollWidth") > VIEWPORTS[vp]["width"] + 1:
            problems.append(f"after clicking '{label}': horizontal overflow")
        if page.url != before:  # a control navigated: come back for the rest
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(500)
            except PWError:
                break
    ctx.close()
    return [f"[{clicked} clicks] {p}" for p in dict.fromkeys(problems)]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = (args[0] if args else "http://127.0.0.1:8777").rstrip("/")
    html = "--html" in sys.argv or base.startswith("http://127.0.0.1")
    pages = PAGES
    for a in sys.argv[1:]:
        if a.startswith("--pages="):
            pages = a.split("=", 1)[1].split(",")
    report: dict[str, list[str]] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        for path in pages:
            for vp in VIEWPORTS:
                for theme in ("dark", "light"):
                    found = audit_page(browser, base, path, vp, theme, html)
                    key = f"{path} {vp} {theme}"
                    report[key] = found
                    print(f"{key}: {'ok' if not found else '; '.join(found)}", flush=True)
        browser.close()
    bad = {k: v for k, v in report.items() if v}
    print(json.dumps({"checked": len(report), "with_problems": len(bad)}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
