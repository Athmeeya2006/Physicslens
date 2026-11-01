"""Smoke test of the Solar System Lab page in a real browser, against a mocked API (no server, no spend).

  python scripts/lab_smoke.py [site]      site defaults to http://127.0.0.1:8777 (a static build being served)

Loads /sky with the analysis server "online" (every /api/sky call answered here), picks Halley with Jupiter and
Saturn, switches Jupiter off, and checks that the equation panel shows the law the API returned. Then loads it with no
server and checks the offline message. Needs Playwright and Chrome; exits non-zero on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import Route, sync_playwright

SITE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777"
ROOT = Path(__file__).resolve().parents[1]
API = "http://lab-api.test"
LAW = {"source": "simulation, no pixels", "status": "accepted", "reason": "Kepler forecasts held-out motion within 0.1%.",
       "law": {"title": "Inverse-square gravity (Kepler)", "latex": r"\ddot x = -\mu x/r^3"},
       "parameters": [{"name": "mu", "value": 2.959e-4, "std": 1e-9, "ci95": [2.95e-4, 2.97e-4]}],
       "perturbation": "No perturbation needed: the residuals look like noise."}


def main() -> int:
    seen: list[str] = []
    bundle = json.loads((ROOT / "frontend" / "public" / "sky" / "halley_with_jupiter_saturn.json").read_text())
    index = json.loads((ROOT / "frontend" / "public" / "sky" / "index.json").read_text())

    def api(route: Route) -> None:
        path = route.request.url.split(API, 1)[1]
        seen.append(path)
        if path.startswith("/api/health"):
            route.fulfill(json={"status": "ok", "audience": "judge", "can_upload": False, "can_go_live": False,
                                "open_access": True, "mock_mode": False})
        elif path.startswith("/api/sky/scenarios"):
            route.fulfill(json=index)
        elif path.startswith("/api/sky/simulate"):
            body = json.loads(route.request.post_data or "{}")
            changed = {**bundle, "bodies": [b for b in bundle["bodies"] if b["name"] not in body.get("disabled", [])]}
            route.fulfill(json=changed)
        elif path.startswith("/api/sky/discover"):
            route.fulfill(json=LAW)
        else:
            route.fulfill(status=404, json={"detail": "not mocked"})

    failures = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True)
        page = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("**/backend.json", lambda r: r.fulfill(json={"api_base": API}))
        page.route(f"{API}/**", api)
        page.goto(f"{SITE}/sky.html#halley_with_jupiter_saturn", wait_until="networkidle")
        page.wait_for_timeout(1500)
        page.get_by_label("Jupiter", exact=True).first.uncheck()
        try:
            page.get_by_text("Inverse-square gravity (Kepler)").first.wait_for(timeout=15000)
        except Exception:  # noqa: BLE001  reported below with what the page did
            failures.append("the equation panel never showed the law")
        if not any(p.startswith("/api/sky/discover") for p in seen):
            failures.append("no discover call was made after the change")
        offline = browser.new_context().new_page()
        offline.goto(f"{SITE}/sky.html#halley_sun_only", wait_until="networkidle")
        offline.wait_for_timeout(1500)
        if not offline.get_by_text("analysis server is offline").count():
            failures.append("offline: no message that the law cannot be found")
        failures += [f"page error: {e}" for e in errors]
        browser.close()
    print("calls:", sorted({s.split("?")[0] for s in seen}))
    print("OK" if not failures else "FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
