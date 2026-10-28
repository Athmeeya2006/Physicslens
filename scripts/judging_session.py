"""Start or stop a judging session: the server runs on a Kaggle GPU notebook, the public site points at it.

  python scripts/judging_session.py start [--hours 6]   upload the code, start the notebook, wait for its tunnel,
                                                       then point the site's /backend.json at it and redeploy
  python scripts/judging_session.py connect              after starting the notebook from Kaggle's editor (Save
                                                       Version > Save & Run All), wait for its tunnel and point
                                                       the site at it
  python scripts/judging_session.py stop                 point the site back to gallery only (the notebook ends
                                                       itself after --hours)

Kaggle attaches notebook Secrets only to runs started from its own editor, not to versions pushed from the CLI,
so a server that needs the keys is started there and connected with `connect`; `start` suits keyless tests.

Needs the Kaggle CLI (pip install kaggle, token in ~/.kaggle) and the Vercel CLI logged in, the deploy folder
var/site/physicslens linked to the Vercel project, and two Kaggle Secrets attached to the notebook:
MODEL_API_KEY and DEMO_ACCESS_CODE. Only the backend address is published; no key leaves Kaggle.
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
    """The notebook runs the committed code, never local uncommitted changes."""
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


def push_kernel(hours: float) -> None:
    with tempfile.TemporaryDirectory() as d:
        k = Path(d)
        code = (ROOT / "scripts" / "kaggle" / "serve_notebook.py").read_text()
        code = re.sub(r"^SERVE_HOURS = .*$", f"SERVE_HOURS = {hours:.2f}", code, count=1, flags=re.MULTILINE)
        (k / "serve_notebook.py").write_text(code)
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": KERNEL, "title": "physicslens-judging-server", "code_file": "serve_notebook.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [],
            "model_sources": []}))
        print(run(KAGGLE, "kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4").strip())


def wait_for_url(minutes: float = 30) -> str:
    """Follow the notebook's log until it prints its tunnel address (setup takes about 5 to 8 minutes)."""
    proc = subprocess.Popen([KAGGLE, "kernels", "logs", "-f", KERNEL], stdout=subprocess.PIPE,
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
    sys.exit("No tunnel address yet; check the notebook's log on Kaggle.")


def point_site(api_base: str) -> None:
    if not (SITE_DIR / ".vercel").is_dir():
        sys.exit(f"{SITE_DIR} is not linked to the Vercel project.")
    (SITE_DIR / "backend.json").write_text(json.dumps({"api_base": api_base}))
    if shutil.which("vercel") is None:
        sys.exit("The Vercel CLI is not installed.")
    subprocess.run(["vercel", "deploy", "--prod", "--yes"], cwd=SITE_DIR, check=True, capture_output=True, timeout=600)
    print(f"site now points at: {api_base or 'nothing (gallery only)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--hours", type=float, default=6.0)
    s.add_argument("--skip-upload", action="store_true", help="reuse the code already uploaded")
    sub.add_parser("stop")
    sub.add_parser("connect")
    a = ap.parse_args()
    if a.cmd == "stop":
        point_site("")
        return
    if a.cmd == "connect":
        url = wait_for_url()
        print(f"server: {url}")
        point_site(url)
        return
    if not a.skip_upload:
        upload_code()
    push_kernel(a.hours)
    url = wait_for_url()
    print(f"server: {url}")
    point_site(url)


if __name__ == "__main__":
    main()
