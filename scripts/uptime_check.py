"""Check the judging server the way a judge reaches it: through the public site's /backend.json.

  python scripts/uptime_check.py              one check, exit code 0 if up
  python scripts/uptime_check.py --watch 300  check every 300 s and print only changes (Ctrl+C to stop)

Makes no model calls and spends nothing: it reads /backend.json, then the server's /api/health.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request

SITE = "https://physicslens.vercel.app"


def fetch(url: str, timeout: float = 15) -> dict:
    if shutil.which("curl"):
        # Cloudflare's DNS directly: a local resolver may still cache a fresh tunnel name as "not found".
        r = subprocess.run(["curl", "-sf", "--max-time", str(timeout), "--doh-url", "https://1.1.1.1/dns-query",
                            "-H", "Cache-Control: no-cache", url], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise OSError(f"curl exit {r.returncode}")
        return json.loads(r.stdout)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache"}), timeout=timeout) as r:
        return json.loads(r.read())


def check() -> tuple[bool, str]:
    try:
        api = fetch(f"{SITE}/backend.json").get("api_base", "")
    except (OSError, ValueError) as e:
        return False, f"site unreachable: {e}"
    if not api:
        return False, "site points at no server (gallery only)"
    t = time.time()
    try:
        h = fetch(f"{api}/api/health")
    except (OSError, ValueError) as e:
        return False, f"server {api} not answering: {e}"
    reasoning = (h.get("reasoning") or {}).get("model", "unknown")
    ok = h.get("status") == "ok" and not h.get("mock_mode") and h.get("app_env") in ("demo", "prod")
    note = "" if ok else " (WRONG MODE: expected demo or prod without mocks)"
    return ok, f"up: {api} env={h.get('app_env')} reasoning={reasoning} {1000 * (time.time() - t):.0f} ms{note}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="repeat, printing only when the state changes")
    a = ap.parse_args()
    if not a.watch:
        ok, msg = check()
        print(msg)
        sys.exit(0 if ok else 1)
    last = None
    while True:
        ok, msg = check()
        if ok != last:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {'UP  ' if ok else 'DOWN'} {msg}", flush=True)
            last = ok
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
