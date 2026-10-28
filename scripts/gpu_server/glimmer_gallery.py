# PhysicsLens on open weights: every reasoning step on Muse Glimmer 30B (Q4_K_M + vision) served by llama.cpp
# on this machine's GPUs. Tracking results (SAM 3.1) arrive as cached responses; this machine has no Meta key
# and the Meta provider is cache-only, so nothing here can call or bill a paid API.
import glob, os, secrets, shutil, subprocess, sys, tarfile, time, urllib.request

T0 = time.time()
def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# Kaggle unpacks uploaded archives; accept either form.
code_tar = find("physicslens-code.tar.gz")
code_dir = None if code_tar else os.path.dirname(os.path.dirname(find("backend/requirements.txt")))
if find("bundles.tar.gz"):
    with tarfile.open(find("bundles.tar.gz")) as t:
        t.extractall("/tmp", filter="data")
    bundles = "/tmp/bundles"
else:
    bundles = os.path.dirname(os.path.dirname(find("info.json")))
print("code:", code_tar or code_dir, "bundles:", bundles, sorted(os.listdir(bundles)), flush=True)

# --- llama.cpp (prebuilt CUDA release, >= b10353 for Muse Glimmer) and the official GGUF weights ---
B = "b11936"
os.chdir("/tmp")
sh(f"curl -sSL -o llama.tgz https://github.com/ggml-org/llama.cpp/releases/download/{B}/llama-{B}-bin-ubuntu-cuda-12.8-x64.tar.gz")
sh(f"curl -sSL -o cudart.tgz https://github.com/ggml-org/llama.cpp/releases/download/{B}/cudart-llama-{B}-bin-ubuntu-cuda-12.8-x64.tar.gz")
sh("mkdir -p llama && tar xzf llama.tgz -C llama && tar xzf cudart.tgz -C llama")
server = subprocess.run("find /tmp/llama -name llama-server -type f | head -1", shell=True, capture_output=True, text=True).stdout.strip()
libdirs = subprocess.run("find /tmp/llama -name '*.so*' -printf '%h\n' | sort -u", shell=True, capture_output=True, text=True).stdout.split()
llama_env = dict(os.environ, LD_LIBRARY_PATH=":".join(libdirs + [os.environ.get("LD_LIBRARY_PATH", "")]))
sh("pip install -q -U huggingface_hub hf_transfer")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download
REPO = "meta-models/Muse-Glimmer-30B-GGUF"
model = hf_hub_download(REPO, "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", local_dir="/tmp/glimmer")
mmproj = hf_hub_download(REPO, "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", local_dir="/tmp/glimmer")
key = secrets.token_hex(16)
os.makedirs("/kaggle/working/out", exist_ok=True)
log = open("/kaggle/working/out/llama-server.log", "w")
proc = subprocess.Popen([server, "-m", model, "--mmproj", mmproj, "-a", "muse-glimmer-30b", "-ngl", "99",
                         "-c", "32768", "-np", "1", "--host", "127.0.0.1", "--port", "8080", "--api-key", key,
                         "--jinja", "--temp", "1.0", "--top-p", "0.95", "--top-k", "64"],
                        stdout=log, stderr=log, env=llama_env)
for _ in range(900):
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/health"), timeout=2)
        break
    except Exception:
        if proc.poll() is not None:
            sys.exit("llama-server exited; see llama-server.log")
        time.sleep(2)
print(f"Glimmer serving after {time.time() - T0:.0f}s", flush=True)

# --- PhysicsLens from the committed source ---
src = "/tmp/physicslens"
if code_tar:
    os.makedirs(src, exist_ok=True)
    with tarfile.open(code_tar) as t:
        t.extractall(src, filter="data")
else:
    shutil.copytree(code_dir, src)  # inputs are read-only; config/ and var/ must be writable
req = [l for l in open(f"{src}/backend/requirements.txt").read().splitlines()
       if l and not l.startswith("#") and not l.split("==")[0].split("[")[0] in ("fastapi", "uvicorn", "pytest", "ruff", "python-multipart", "httpx")]
try:
    sh("pip install -q " + " ".join(f"'{r}'" for r in req))
except subprocess.CalledProcessError:
    sh("pip install -q " + " ".join(f"'{r.split('==')[0]}'" for r in req))
shutil.copy(f"{src}/config/budget.example.yaml", f"{src}/config/budget.yaml")
os.makedirs(f"{src}/var", exist_ok=True)
cache_in = find("cache_in.db")
if cache_in:
    shutil.copy(cache_in, f"{src}/var/cache.db")
env = dict(os.environ, APP_ENV="dev", PYTHONPATH=f"{src}/backend", REASONING_PROVIDER="glimmer",
           PHYSICSLENS_CACHE_ONLY="meta_api", GLIMMER_BASE_URL="http://127.0.0.1:8080/v1", GLIMMER_API_KEY=key,
           MODEL_API_KEY="")
subprocess.run([sys.executable, f"{src}/scripts/glimmer_batch.py", "run", "--bundles", bundles, "--out", "/kaggle/working/out"],
               env=env, check=False)
proc.terminate()
print(f"finished after {time.time() - T0:.0f}s", flush=True)
