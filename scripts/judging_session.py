"""Start or stop a judging session: the server runs on a Kaggle machine, the public site points at it.

  python scripts/judging_session.py start [--hours 11.5] [--gpu]
                                                       upload the code, start the server (CPU by default:
                                                       reasoning on Muse Spark Open via Baseten; --gpu adds
                                                       Muse Glimmer as the fallback), wait for its tunnel,
                                                       then point the site's /backend.json at it and redeploy
  python scripts/judging_session.py connect              after starting the server from Kaggle's editor (Save
                                                       Version > Save & Run All), wait for its tunnel and point
                                                       the site at it
  python scripts/judging_session.py connect --url URL    point the site at a server address read from its log
                                                       page (an editor run's log is only readable from the
                                                       CLI after the run ends)
  python scripts/judging_session.py stop                 point the site back to gallery only (the server ends
                                                       itself after --hours)
  python scripts/judging_session.py pointer GIST_ID  the site finds the server through a public gist that each server
                                                       updates itself (Kaggle Secrets GIST_TOKEN and GIST_ID): needed
                                                       for unattended, scheduled sessions
  python scripts/judging_session.py start --slot b      a second notebook (physicslens-judging-server-b), scheduled
                                                       12 hours after the first so the two cover the whole day
  python scripts/judging_session.py schedule 2026-11-05T14:00Z
                                                       show the next session's start on the site meanwhile

Kaggle attaches Secrets only to runs started from its own editor, not to versions pushed from the CLI,
so a server that needs the keys is started there and connected with `connect`; `start` suits keyless tests.

Needs the Kaggle CLI (pip install kaggle, token in ~/.kaggle) and the Vercel CLI logged in, the deploy folder
var/site/physicslens linked to the Vercel project, and three Kaggle Secrets attached to the server:
BASETEN_API_KEY, MODEL_API_KEY and DEMO_ACCESS_CODE. Only the backend address is published; no key leaves Kaggle.
A laptop can serve instead (scripts/serve_local.py); `connect --url` points the site at either.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "var" / "site" / "physicslens"
KAGGLE = os.environ.get("KAGGLE_BIN", "kaggle")
OWNER = os.environ.get("KAGGLE_OWNER", "athmeeyamkashyap")
DATASET = f"{OWNER}/physicslens-server-code"  # its own dataset: a new version replaces all files
KERNEL = f"{OWNER}/physicslens-judging-server"


def run(*cmd: str, timeout: float = 600) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout).stdout


def upload_code() -> None:
    """The server runs the committed code, never local uncommitted changes."""
    with tempfile.TemporaryDirectory() as d:
        data = Path(d)
        subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(data / "physicslens-code.tar.gz"), "HEAD"],
                       cwd=ROOT, check=True)
        (data / "dataset-metadata.json").write_text(json.dumps(
            {"title": "physicslens-server-code", "id": DATASET, "licenses": [{"name": "other"}]}))
        exists = subprocess.run([KAGGLE, "datasets", "status", DATASET], capture_output=True, text=True, check=False).returncode == 0
        if exists:
            run(KAGGLE, "datasets", "version", "-p", str(data), "-m", "judging server code")
        else:
            run(KAGGLE, "datasets", "create", "-p", str(data))  # private by default
    for _ in range(60):
        if run(KAGGLE, "datasets", "status", DATASET).strip().endswith("ready"):
            return
        time.sleep(10)
    sys.exit("The Kaggle dataset did not become ready.")


def push_kernel(hours: float, gpu: bool, slot: str = "") -> None:
    kernel = KERNEL + (f"-{slot}" if slot else "")
    with tempfile.TemporaryDirectory() as d:
        k = Path(d)
        code = (ROOT / "scripts" / "gpu_server" / "serve.py").read_text()
        code = re.sub(r"^SERVE_HOURS = .*$", f"SERVE_HOURS = {hours:.2f}", code, count=1, flags=re.MULTILINE)
        code = re.sub(r"^SERVE_GPU = .*$", f"SERVE_GPU = {gpu}", code, count=1, flags=re.MULTILINE)
        (k / "serve.py").write_text(code)
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel, "title": kernel.split("/", 1)[1], "code_file": "serve.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [],
            "model_sources": []}))
        accel = ["--accelerator", "NvidiaTeslaT4"] if gpu else []
        print(run(KAGGLE, "kernels", "push", "-p", str(k), *accel).strip())


def wait_for_url(minutes: float = 30, kernel: str = KERNEL) -> str:
    """Follow the server's log until it prints its tunnel address (setup takes about 5 to 8 minutes)."""
    proc = subprocess.Popen([KAGGLE, "kernels", "logs", "-f", kernel], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + minutes * 60
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = re.search(r"PHYSICSLENS_URL=(https://[a-z0-9-]+\.trycloudflare\.com)", line)
            if m:
                return m.group(1)
            if "secrets:" in line or "serving after" in line or "backend up" in line:
                print("  " + re.sub(r'.*"data":"', "", line).strip().rstrip('"}').replace("\\n", ""))
            if time.time() > deadline:
                break
    finally:
        proc.terminate()
    sys.exit("No tunnel address yet; check the server's log on Kaggle.")


def point_site(api_base: str | None = None, next_session_utc: str | None = None, pointer_gist: str | None = None) -> None:
    """Change what the site's /backend.json says; None keeps the current value of that field."""
    if not (SITE_DIR / ".vercel").is_dir():
        sys.exit(f"{SITE_DIR} is not linked to the Vercel project.")
    path = SITE_DIR / "backend.json"
    cfg = json.loads(path.read_text()) if path.is_file() else {}
    if api_base is not None:
        cfg["api_base"] = api_base
    if next_session_utc is not None:
        cfg["next_session_utc"] = next_session_utc
    if pointer_gist is not None:
        cfg["pointer_gist"] = pointer_gist
    cfg.setdefault("api_base", "")
    path.write_text(json.dumps({k: v for k, v in cfg.items() if v != "" or k == "api_base"}))
    api_base = cfg["api_base"]
    if shutil.which("vercel") is None:
        sys.exit("The Vercel CLI is not installed.")
    subprocess.run(["vercel", "deploy", "--prod", "--yes"], cwd=SITE_DIR, check=True, capture_output=True, timeout=600)
    print(f"site now points at: {api_base or 'nothing (gallery only)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--hours", type=float, default=11.5, help="Kaggle stops any run at 12 h")
    s.add_argument("--slot", default="", help="a second notebook, e.g. b, for back-to-back scheduled sessions")
    s.add_argument("--gpu", action="store_true", help="also serve Muse Glimmer as the open-weight fallback (uses GPU quota)")
    s.add_argument("--skip-upload", action="store_true", help="reuse the code already uploaded")
    sub.add_parser("stop")
    cn = sub.add_parser("connect")
    cn.add_argument("--url", help="the server's https://...trycloudflare.com address, if already known")
    pt = sub.add_parser("pointer", help="let the site read the server address from a gist the server updates itself")
    pt.add_argument("gist_id", help='the gist id, or "none" to stop using it')
    sc = sub.add_parser("schedule", help="show the next session's start time on the site while no server runs")
    sc.add_argument("when", help='ISO time in UTC, e.g. 2026-11-05T14:00Z, or "none" to clear it')
    a = ap.parse_args()
    if a.cmd == "stop":
        point_site("")
        return
    if a.cmd == "pointer":
        if a.gist_id != "none" and not re.fullmatch(r"[0-9a-f]{20,40}", a.gist_id):
            sys.exit("Expected a gist id (the hex string at the end of its URL).")
        # The fixed address (if any) stays: the site tries it first and follows the gist once it stops answering.
        point_site(None, pointer_gist="" if a.gist_id == "none" else a.gist_id)
        print("the site now follows the gist" if a.gist_id != "none" else "gist pointer removed")
        return
    if a.cmd == "schedule":
        when = "" if a.when == "none" else datetime.fromisoformat(a.when).astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")
        point_site(next_session_utc=when)
        print(f"next session shown on the site: {when or 'none'}")
        return
    if a.cmd == "connect":
        if a.url and not re.fullmatch(r"https://[a-z0-9-]+\.trycloudflare\.com", a.url.rstrip("/")):
            sys.exit("Expected an address like https://some-words.trycloudflare.com")
        url = a.url.rstrip("/") if a.url else wait_for_url()
        print(f"server: {url}")
        point_site(url)
        return
    if not a.skip_upload:
        upload_code()
    push_kernel(min(a.hours, 11.75), a.gpu, a.slot)
    url = wait_for_url(kernel=KERNEL + (f"-{a.slot}" if a.slot else ""))
    print(f"server: {url}")
    point_site(url)


if __name__ == "__main__":
    main()
