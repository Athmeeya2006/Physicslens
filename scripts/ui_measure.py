"""How long and how wordy each page is: screens of scrolling and visible words, before and after a layout change.

  python3.12 scripts/ui_measure.py LABEL [BASE_URL]     # BASE_URL defaults to the static build on 127.0.0.1:8777

Opens each page at desktop (1440x900) and phone (390x844) width in the dark and light themes, waits for it to settle,
and records the page height in screens and the number of words a reader can see (hidden and collapsed text not
counted). Writes ../physicslens-ui/LABEL/measure.json and a full-page screenshot per page, viewport and theme, outside
the repository. Free: it never uploads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {"home": "index.html", "cases": "cases.html", "pendulum": "index.html?run=gallery-pendulum",
         "paperbag": "index.html?run=gallery-paperbag", "timing": "timing.html", "fireball": "fireball.html",
         "sun": "sun.html", "satellites": "satellites.html", "predictions": "predictions.html"}
VIEWPORTS = {"desktop": (1440, 900), "phone": (390, 844)}
OUT = Path(__file__).resolve().parents[2] / "physicslens-ui"

WORDS = """() => {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n = 0;
  while (walker.nextNode()) {
    const el = walker.currentNode.parentElement;
    if (!el || !el.checkVisibility || !el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue;
    if (el.closest('details:not([open]) > :not(summary)')) continue;
    n += (walker.currentNode.textContent.match(/[A-Za-z0-9][^\\s]*/g) || []).length;
  }
  return n;
}"""


def main() -> None:
    label = sys.argv[1]
    base = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8777").rstrip("/")
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for vp, (w, h) in VIEWPORTS.items():
            for theme in ("dark", "light"):
                ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=theme)
                # The site's own switch: light is the default, dark is data-theme="dark", remembered in localStorage.
                ctx.add_init_script(f"try {{ localStorage.setItem('physicslens-theme', '{theme}'); }} catch (e) {{}}"
                                    + ("document.documentElement.dataset.theme = 'dark';" if theme == "dark" else ""))
                page = ctx.new_page()
                for name, path in PAGES.items():
                    page.goto(f"{base}/{path}", wait_until="load")
                    page.wait_for_timeout(2500)
                    height = page.evaluate("document.documentElement.scrollHeight")
                    first = page.evaluate("""() => { let n = 0; const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                      while (w.nextNode()) { const r = w.currentNode.parentElement?.getBoundingClientRect();
                        if (r && r.top < innerHeight && r.bottom > 0 && r.width > 0) n += (w.currentNode.textContent.match(/[A-Za-z0-9][^\\s]*/g) || []).length; }
                      return n; }""")
                    rows.append({"page": name, "viewport": vp, "theme": theme, "screens": round(height / h, 1),
                                 "words": page.evaluate(WORDS), "first_screen_words": first})
                    page.screenshot(path=str(out / f"{name}-{vp}-{theme}.png"), full_page=True)
                ctx.close()
        browser.close()
    (out / "measure.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        if r["theme"] == "dark":
            print(f"{r['page']:12s} {r['viewport']:8s} {r['screens']:5.1f} screens {r['words']:6d} words, first screen {r['first_screen_words']}")


if __name__ == "__main__":
    main()
