# ruff: noqa: SIM115, PLW1510, BLE001, S110
# A long-running setup script: log files stay open for the lifetime of the servers they capture, probes may fail
# while a server starts, and an unavailable secrets service just means "no secret".
# PhysicsLens judging server on a Kaggle machine, reachable from the public site through a Cloudflare quick tunnel.
#
# - SERVE_GPU = False (default): CPU only. Reasoning on Muse Spark Open (open weights) served by Baseten; fitting,
#   MuJoCo and ffmpeg need no GPU. SERVE_GPU = True also starts Muse Glimmer (llama.cpp) on the GPUs as the
#   open-weight fallback, or as the only reasoning model if no Baseten key is attached.
# - Muse Glimmer listens on 127.0.0.1 only and is never exposed; only the backend is tunnelled.
# - Keys come from Kaggle Secrets (Add-ons > Secrets): BASETEN_API_KEY (Muse Spark Open), MODEL_API_KEY (Meta Model
#   API, for SAM 3.1 and voice) and DEMO_ACCESS_CODE (what judges use). Nothing secret is printed or written.
# - The backend runs with APP_ENV=demo: reasoning only on open-weight models, judges need the access code,
#   public visitors get the gallery only. Spending caps come from a secret gist (PRIVATE_GIST_ID), never from this
#   public file; the same gist carries the judges' running spend from one daily session to the next.
# - scripts/judging_session.py pushes this script, reads the tunnel address from its log and points the site at it.
import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request

SERVE_HOURS = 6.0  # replaced by judging_session.py; the session ends itself (Kaggle stops any run at 12 h)
SERVE_GPU = False  # replaced by judging_session.py
PRIVATE_FILE = "physicslens-private.json"  # in the secret gist: {"session_budget": {...}, "judge_spent": {...}}
SITE = "https://physicslens.vercel.app"
T0 = time.time()


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def find(name: str) -> str | None:
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


def wait_http(url: str, seconds: int, proc: subprocess.Popen | None = None, headers: dict | None = None) -> bool:
    for _ in range(seconds // 2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(2)
    return False


# The site finds a server that restarts on a schedule through a public gist holding its current address. The token
# (Kaggle Secret GIST_TOKEN, a GitHub token limited to gists) can only edit gists; GIST_ID names the one to write.
POINTER_FILE = "physicslens-backend.json"


def announce(api_base: str, only_if_mine: str | None = None) -> None:
    token, gist_id = secret("GIST_TOKEN"), secret("GIST_ID")
    if not token or not gist_id:
        print("pointer: GIST_TOKEN/GIST_ID not set; point the site with judging_session.py connect", flush=True)
        return
    api = f"https://api.github.com/gists/{gist_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "User-Agent": "physicslens-server"}
    try:
        if only_if_mine is not None:
            # At shutdown, clear the pointer only if it still names this session (a newer one may have taken over).
            with urllib.request.urlopen(urllib.request.Request(api, headers=headers), timeout=20) as r:
                current = json.loads(json.loads(r.read())["files"].get(POINTER_FILE, {}).get("content") or "{}")
            if current.get("api_base") != only_if_mine:
                return
        until = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 + SERVE_HOURS * 3600))
        body = {"files": {POINTER_FILE: {"content": json.dumps({"api_base": api_base, "until_utc": until if api_base else None,
                                                                 "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})}}}
        req = urllib.request.Request(api, data=json.dumps(body).encode(), method="PATCH",
                                     headers={**headers, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"pointer: published {'address' if api_base else 'gallery only'} (HTTP {r.status})", flush=True)
    except Exception as e:
        print(f"pointer: could not update the gist ({type(e).__name__})", flush=True)  # never the token


def _gist(gist_id: str, body: dict | None = None) -> dict:
    token = secret("GIST_TOKEN")
    req = urllib.request.Request(f"https://api.github.com/gists/{gist_id}", method="PATCH" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                          "User-Agent": "physicslens-server", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def read_private() -> dict:
    """The session budget and the judges' spend so far. Missing means no caps known, so the server must not serve."""
    gist_id = secret("PRIVATE_GIST_ID")
    if not gist_id or not secret("GIST_TOKEN"):
        return {}
    try:
        return json.loads(_gist(gist_id)["files"][PRIVATE_FILE]["content"])
    except Exception as e:
        print(f"private gist: could not read it ({type(e).__name__})", flush=True)
        return {}


def save_judge_spent(private: dict, ledger_db: str) -> None:
    """Earlier sessions' judge spend plus this session's, written back so tomorrow's session starts from it."""
    import sqlite3
    start = private["session_budget"].get("judge_period_start", "")
    try:
        with sqlite3.connect(ledger_db) as conn:
            rows = conn.execute("SELECT provider, COALESCE(SUM(est_cost_usd), 0) FROM spend WHERE audience = 'judge' "
                                "AND status IN ('committed', 'failed_billed', 'reserved') AND ts >= ? GROUP BY provider",
                                (start,)).fetchall()
    except sqlite3.Error:
        return  # no ledger yet: nothing spent
    before = private.get("judge_spent", {})
    now = {p: round(before.get(p, 0.0) + dict(rows).get(p, 0.0), 6) for p in set(before) | {r[0] for r in rows}}
    if now == private.get("_last_saved"):
        return
    try:
        _gist(secret("PRIVATE_GIST_ID"), {"files": {PRIVATE_FILE: {"content": json.dumps({**{k: v for k, v in private.items()
                                                                                               if not k.startswith("_")},
                                                                                            "judge_spent": now}, indent=1)}}})
        private["_last_saved"] = now
    except Exception as e:
        print(f"private gist: could not save judge spend ({type(e).__name__})", flush=True)


def secret(name: str) -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or ""
    except Exception:
        return ""


baseten_key, model_key, access_code = secret("BASETEN_API_KEY"), secret("MODEL_API_KEY"), secret("DEMO_ACCESS_CODE")
print("secrets: " + ", ".join(f"{n} {'present' if v else 'MISSING'}" for n, v in (
    ("BASETEN_API_KEY", baseten_key), ("MODEL_API_KEY", model_key), ("DEMO_ACCESS_CODE", access_code))), flush=True)
if not baseten_key and not SERVE_GPU:
    # Runs pushed from the command line never get Kaggle Secrets; that is expected, not a failure. Ending cleanly keeps
    # the notebook's history readable: only a run started from the editor or the schedule can serve.
    print("PHYSICSLENS_NOT_SERVING: this run has no secrets (pushed from the command line). Attach BASETEN_API_KEY, "
          "MODEL_API_KEY, DEMO_ACCESS_CODE, GIST_TOKEN and GIST_ID under Add-ons > Secrets, then use Save & Run All "
          "or the schedule.", flush=True)
    sys.exit(0)
os.makedirs("/kaggle/working/logs", exist_ok=True)
private = read_private()
if not private.get("session_budget"):
    # Never serve without known caps (and never fall back to a guess): judges would get an unguarded server.
    print("PHYSICSLENS_NOT_SERVING: no session budget. Add PRIVATE_GIST_ID (a secret gist holding "
          f"{PRIVATE_FILE}) and GIST_TOKEN under Add-ons > Secrets.", flush=True)
    sys.exit(1)
print("private gist: session budget loaded; judge spend so far carried over", flush=True)


def start_glimmer() -> tuple[subprocess.Popen, str]:
    """Muse Glimmer on llama.cpp (prebuilt CUDA release, >= b10353), bound to localhost."""
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    b = "b11936"
    os.chdir("/tmp")
    sh(f"curl -sSL -o llama.tgz https://github.com/ggml-org/llama.cpp/releases/download/{b}/llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz")
    sh(f"curl -sSL -o cudart.tgz https://github.com/ggml-org/llama.cpp/releases/download/{b}/cudart-llama-{b}-bin-ubuntu-cuda-12.8-x64.tar.gz")
    sh("mkdir -p llama && tar xzf llama.tgz -C llama && tar xzf cudart.tgz -C llama")
    server = subprocess.run("find /tmp/llama -name llama-server -type f | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    libdirs = subprocess.run("find /tmp/llama -name '*.so*' -printf '%h\\n' | sort -u", shell=True, capture_output=True, text=True).stdout.split()
    llama_env = dict(os.environ, LD_LIBRARY_PATH=":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    sh("pip install -q -U huggingface_hub hf_transfer")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    repo = "meta-models/Muse-Glimmer-30B-GGUF"
    revision = "b44f1072cf75e918a1b82a3a17b45c4389b076e7"  # pinned: the exact files listed in OPEN_WEIGHTS.md
    weights = {f: hf_hub_download(repo, f, local_dir="/tmp/glimmer", revision=revision) for f in (
        "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", "dflash-Muse-Glimmer-30B-Q4_K_M.gguf")}
    key = secrets.token_hex(16)
    proc = subprocess.Popen(
        [server, "-m", weights["Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"],
         "--mmproj", weights["mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"],
         "-md", weights["dflash-Muse-Glimmer-30B-Q4_K_M.gguf"], "-ngld", "99",  # speculative decoding: same outputs, faster
         "-a", "muse-glimmer-30b", "-ngl", "99", "-c", "65536", "-np", "2",  # two analyses at once, 32k context each
         "--host", "127.0.0.1", "--port", "8080", "--api-key", key,
         "--jinja", "--temp", "1.0", "--top-p", "0.95", "--top-k", "64"],
        stdout=open("/kaggle/working/logs/llama-server.log", "w"), stderr=subprocess.STDOUT, env=llama_env)
    if not wait_http("http://127.0.0.1:8080/health", 900, proc):
        sys.exit("llama-server did not start; see logs/llama-server.log")
    print(f"Muse Glimmer serving after {time.time() - T0:.0f}s", flush=True)
    return proc, key


procs: dict[str, subprocess.Popen] = {}
reasoning_env = {"REASONING_PROVIDER": "baseten", "BASETEN_API_KEY": baseten_key}
if SERVE_GPU:
    procs["glimmer"], glimmer_key = start_glimmer()
    # With a Baseten key Glimmer is the fallback; without one it is the reasoning model.
    reasoning_env = {"REASONING_PROVIDER": "baseten" if baseten_key else "glimmer", "BASETEN_API_KEY": baseten_key,
                     "GLIMMER_BASE_URL": "http://127.0.0.1:8080/v1", "GLIMMER_API_KEY": glimmer_key}
print(f"reasoning: {reasoning_env['REASONING_PROVIDER']}" + (" (Glimmer fallback ready)" if SERVE_GPU and baseten_key else ""),
      flush=True)

# ---------- PhysicsLens backend ----------
src = "/tmp/physicslens"
code_tar = find("physicslens-code.tar.gz")  # Kaggle usually unpacks uploaded archives; accept either form
if code_tar:
    os.makedirs(src)
    sh(f"tar xzf {code_tar} -C {src}")
else:
    shutil.copytree(os.path.dirname(os.path.dirname(find("backend/requirements.txt"))), src)
req = [r for r in open(f"{src}/backend/requirements.txt").read().splitlines()
       if r and not r.startswith("#") and r.split("==")[0] not in ("pytest", "ruff")]
try:
    sh("pip install -q " + " ".join(f"'{r}'" for r in req))
except subprocess.CalledProcessError:
    sh("pip install -q " + " ".join(f"'{r.split('==')[0]}'" for r in req))
if shutil.which("ffmpeg") is None:
    sh("apt-get -qq update && apt-get -qq install -y ffmpeg > /dev/null")
import yaml

with open(f"{src}/config/budget.yaml", "w") as f:
    yaml.safe_dump(private["session_budget"], f)
env = dict(os.environ, PYTHONPATH=f"{src}/backend", APP_ENV="demo", MOCK_MODELS="0", USE_CONTRIBUTOR_MODEL="0",
           **reasoning_env, MODEL_API_KEY=model_key, DEMO_ACCESS_CODE=access_code, PUBLIC_TRY_ENABLED="0", FRONTEND_ORIGINS=SITE,
           JUDGE_SPENT_BEFORE=json.dumps(private.get("judge_spent", {})),
           # "open_access": true in the private gist gives every visitor the full site with no code; TRUST_PROXY lets
           # the per-visitor allowance read Cloudflare's visitor address (through the tunnel, everyone is 127.0.0.1).
           OPEN_ACCESS="1" if private.get("open_access") or private["session_budget"].get("open_access") else "0",
           TRUST_PROXY="1")
procs["backend"] = backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "physicslens.app:app", "--host", "127.0.0.1", "--port", "8000"],
                           cwd=src, env=env, stdout=open("/kaggle/working/logs/backend.log", "w"), stderr=subprocess.STDOUT)
if not wait_http("http://127.0.0.1:8000/api/health", 180, backend):
    sys.exit("The backend did not start; see logs/backend.log")
print(f"backend up after {time.time() - T0:.0f}s", flush=True)

# ---------- Cloudflare quick tunnel to the backend only ----------
sh("curl -sSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
   " && chmod +x /tmp/cloudflared")
tunnel_log = "/kaggle/working/logs/tunnel.log"
procs["tunnel"] = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"],
                          stdout=open(tunnel_log, "w"), stderr=subprocess.STDOUT)
url = None
for _ in range(90):
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(tunnel_log).read())
    if m:
        url = m.group(0)
        break
    time.sleep(2)
if url is None or not wait_http(f"{url}/api/health", 120):
    sys.exit("The tunnel did not come up; see logs/tunnel.log")
print(f"PHYSICSLENS_URL={url}", flush=True)
announce(url)

# ---------- Serve until the time is up, checking every few minutes ----------
while time.time() - T0 < SERVE_HOURS * 3600:
    time.sleep(300)
    dead = [n for n, p in procs.items() if p.poll() is not None]
    if dead or not wait_http(f"{url}/api/health", 30):
        print(f"PHYSICSLENS_DOWN {dead or ['tunnel health check']}", flush=True)
        break
    save_judge_spent(private, f"{src}/var/ledger.db")
    print(f"alive {(time.time() - T0) / 3600:.1f} h", flush=True)
print("PHYSICSLENS_STOPPING", flush=True)
save_judge_spent(private, f"{src}/var/ledger.db")
announce("", only_if_mine=url)
for p in procs.values():
    p.terminate()
