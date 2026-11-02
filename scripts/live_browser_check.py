"""Live mode in a real browser, for every motion type it offers, with Chrome's fake camera.

  python scripts/live_browser_check.py make           # synthetic camera videos (y4m) under var/live_check/
  python3.12 scripts/live_browser_check.py run URL      # URL: the site (e.g. http://127.0.0.1:8777, its live.html)
  DEMO_ACCESS_CODE=... python3.12 scripts/live_browser_check.py run https://physicslens.vercel.app   # production (paid: SAM)

Each video shows one bright ball on a textured background moving as one family does (a swing, a spring, a throw, a
slide). The check starts live analysis, taps the ball where it is on screen, picks the motion type, and waits for a
fitted law of that family, reporting what the page showed. Run against a local backend with MOCK_MODELS=1 it costs
nothing: live tracking after a tap is local, and the reasoning model's answers are the test doubles.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var" / "live_check"
W, H, FPS, SECONDS = 640, 480, 25, 40
PX_PER_M = 400.0  # how the synthetic scene maps metres to pixels


def ball_xy(family: str, t: float) -> tuple[float, float]:
    """Where the ball is (pixels) at time t, for each family."""
    if family == "pendulum":
        L, th0, g = 0.6, 0.35, 9.81
        th = th0 * math.exp(-0.03 * t) * math.cos(math.sqrt(g / L) * t)
        return W / 2 + PX_PER_M * L * math.sin(th), 60 + PX_PER_M * L * math.cos(th)
    if family == "oscillator":
        return W / 2, H / 2 + 120 * math.exp(-0.05 * t) * math.cos(2 * math.pi * 0.7 * t)
    if family == "projectile":
        # One throw across the frame (slow, as a phone's slow motion shows it), then the ball lies where it landed.
        tt = min(max(t - 1.0, 0.0), 6.0)
        return 40 + 90 * tt, 400 - (110 * tt - 18.3 * tt * tt)
    # One slide that slows by friction to a stop, then rests.
    tt = max(t - 1.0, 0.0)
    v0, a = 1.0, 0.2
    s = v0 * tt - 0.5 * a * tt * tt if tt < v0 / a else v0 * v0 / (2 * a)
    return 40 + PX_PER_M * 0.22 * s * 2.5, H * 0.7


def make() -> None:
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    bg = (rng.integers(20, 70, (H, W, 3))).astype(np.uint8)  # texture, so a tapped patch has detail
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    families = sys.argv[2:] or ["pendulum", "oscillator", "projectile", "line"]
    for family in families:
        raw = OUT / f"{family}.mp4"
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for k in range(FPS * SECONDS):
            f = bg.copy()
            x, y = ball_xy(family, k / FPS)
            if family == "pendulum":
                cv2.line(f, (W // 2, 60), (int(x), int(y)), (200, 200, 200), 2)
            cv2.circle(f, (int(x), int(y)), 16, (40, 200, 250), -1)
            cv2.circle(f, (int(x), int(y)), 16, (20, 90, 160), 2)
            vw.write(f)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-pix_fmt", "yuv420p", str(OUT / f"{family}.y4m")], check=True)
        print(family, (OUT / f"{family}.y4m").stat().st_size // 1024, "KB")


LABEL = {"pendulum": "Swinging", "oscillator": "On a spring", "projectile": "Thrown or falling", "line": "Rolling or sliding"}


def run(base: str) -> None:
    from playwright.sync_api import sync_playwright
    results = {}
    with sync_playwright() as p:
        for family in [f for f in LABEL if not sys.argv[3:] or f in sys.argv[3:]]:
            browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=[
                "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-video-capture={OUT / f'{family}.y4m'}"])
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            import os
            from urllib.parse import quote
            local = base.startswith(("http://127.0.0.1", "http://localhost"))
            url = f"{base}/live.html" if local else f"{base}/live"
            code = os.environ.get("DEMO_ACCESS_CODE")  # the judges' code, for a production check (never printed)
            if code and not local:
                url += f"?code={quote(code)}"
            page.goto(url, wait_until="load")
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="Start live analysis").wait_for(state="visible")
            for _ in range(120):
                if page.get_by_role("button", name="Start live analysis").is_enabled():
                    break
                time.sleep(0.5)
            page.get_by_role("button", name="Start live analysis").click()
            page.wait_for_timeout(2500)
            # Tap the ball where the video shows it now (the fake camera loops the file from the start).
            vt = page.evaluate("() => document.querySelector('video').currentTime")
            x, y = ball_xy(family, vt + 0.05)
            box = page.locator("canvas").first.bounding_box()
            page.mouse.click(box["x"] + x / W * box["width"], box["y"] + y / H * box["height"])
            page.wait_for_timeout(500)
            chip = page.get_by_role("button", name=LABEL[family])
            if chip.count():
                chip.first.click()
            found, text = None, ""
            for _ in range(90):
                page.wait_for_timeout(1000)
                text = page.locator("main").inner_text()
                eq = page.locator(".equation")
                if eq.count():
                    found = text
                    break
            dark = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
            page.screenshot(path=str(OUT / f"{family}.png"))
            results[family] = {"law_shown": found is not None, "errors": errors,
                               "excerpt": " ".join((found or text).split())[:300], "body_bg": dark}
            print(family, json.dumps(results[family])[:400], flush=True)
            browser.close()
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    sys.exit(0 if all(r["law_shown"] and not r["errors"] for r in results.values()) else 1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["make"]:
        make()
    else:
        run(sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "http://127.0.0.1:8777")
