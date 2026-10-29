"""Serve the judging backend from this machine and point the public site at it.

  python scripts/serve_local.py              start the backend (APP_ENV=demo) and a Cloudflare quick tunnel,
                                             point the site's /backend.json at the tunnel, and keep both alive
  python scripts/serve_local.py --no-site    same, without touching the site (for a private test)

Run it under `systemd-inhibit --what=sleep:idle --why="PhysicsLens judging" python scripts/serve_local.py` so the
laptop does not suspend. Keys come from .env (BASETEN_API_KEY, MODEL_API_KEY, DEMO_ACCESS_CODE) and stay on this
machine; only the backend address is published. Spending is capped by config/budget.yaml and the persistent
ledger in var/, like every other run. Ctrl+C stops the server and points the site back to gallery only.

A quick tunnel gets a new address each time it starts, so when the tunnel dies it is restarted and the site is
re-pointed. If the backend itself dies it is restarted too.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judging_session import point_site

SITE = "https://physicslens.vercel.app"
PORT = 8000
LOGS = ROOT / "var" / "serve_local"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
CHECK_SECONDS = 60


def healthy(url: str, timeout: float = 10) -> bool:
    if "trycloudflare.com" in url and shutil.which("curl"):
        # A fresh tunnel name is often cached as "not found" by the local resolver for minutes, while the rest of
        # the internet resolves it at once; ask Cloudflare's DNS directly, as a judge's browser effectively would.
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(timeout),
                            "--doh-url", "https://1.1.1.1/dns-query", f"{url}/api/health"],
                           capture_output=True, text=True, check=False)
        return r.stdout.strip() == "200"
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def wait_healthy(url: str, seconds: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if healthy(url, timeout=5):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(2)
    return False


def cloudflared() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    path = ROOT / "var" / "bin" / "cloudflared"
    if not path.exists():
        arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), "amd64")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading cloudflared ...", flush=True)
        part = path.with_suffix(".part")
        url = CLOUDFLARED_URL.format(arch=arch)
        if shutil.which("curl"):  # Python's urllib stalled on GitHub's release redirect on one machine; curl did not
            subprocess.run(["curl", "-fsSL", "--max-time", "300", "-o", str(part), url], check=True)
        else:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
        part.chmod(0o755)
        part.rename(path)  # only a complete download is ever used
    return str(path)


def start_backend() -> subprocess.Popen:
    # Explicit values win over .env (the app loads .env without overriding what is already set).
    env = dict(os.environ, PYTHONPATH=str(ROOT / "backend"), APP_ENV="demo", MOCK_MODELS="0",
               USE_CONTRIBUTOR_MODEL="0", REASONING_PROVIDER=os.environ.get("REASONING_PROVIDER", "baseten"),
               PUBLIC_TRY_ENABLED="0", FRONTEND_ORIGINS=SITE)
    log = open(LOGS / "backend.log", "a")  # noqa: SIM115 (open for the life of the process)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "physicslens.app:app", "--host", "127.0.0.1",
                             "--port", str(PORT)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not wait_healthy(f"http://127.0.0.1:{PORT}", 120, proc):
        sys.exit(f"The backend did not start; see {LOGS / 'backend.log'}")
    return proc


def start_tunnel() -> tuple[subprocess.Popen, str]:
    log_path = LOGS / "tunnel.log"
    log = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen([cloudflared(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                            stdout=log, stderr=subprocess.STDOUT)
    for _ in range(90):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log_path.read_text())
        if m and wait_healthy(m.group(0), 120, proc):
            return proc, m.group(0)
        if proc.poll() is not None:
            break
        time.sleep(2)
    proc.terminate()
    raise RuntimeError(f"The tunnel did not come up; see {log_path}")


def _stop(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def repoint(url: str) -> None:
    """A failed site deploy must not stop the server; the next tunnel change tries again."""
    try:
        point_site(url)
    except (SystemExit, subprocess.SubprocessError) as e:
        print(f"could not point the site at {url or 'gallery only'}: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-site", action="store_true", help="do not point the public site at this server")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    # `kill` should clean up like Ctrl+C does: stop the children and point the site back to gallery only.
    signal.signal(signal.SIGTERM, _stop)

    backend = start_backend()
    print(f"backend up on 127.0.0.1:{PORT}", flush=True)
    tunnel: subprocess.Popen | None = None
    try:
        tunnel, url = start_tunnel()
        print(f"PHYSICSLENS_URL={url}", flush=True)
        if not a.no_site:
            repoint(url)
        while True:
            time.sleep(CHECK_SECONDS)
            if backend.poll() is not None:
                print("backend exited; restarting", flush=True)
                backend = start_backend()
            if tunnel.poll() is not None or not healthy(url):
                print(f"{time.strftime('%H:%M:%S')} tunnel unhealthy; starting a new one", flush=True)
                tunnel.terminate()
                try:
                    tunnel, url = start_tunnel()
                except RuntimeError as e:  # offline for now: try again on the next check
                    print(e, flush=True)
                    continue
                print(f"PHYSICSLENS_URL={url}", flush=True)
                if not a.no_site:
                    repoint(url)
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        # Whatever went wrong, never leave a backend running with nothing pointing at it, or the site at a dead one.
        if tunnel is not None:
            tunnel.terminate()
        backend.terminate()
        if not a.no_site:
            repoint("")


if __name__ == "__main__":
    main()
