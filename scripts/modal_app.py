"""PhysicsLens on Modal: the analysis server on CPU, always reachable, scaled to zero when nobody uses it.

Reasoning runs on Muse Spark Open (open weights) served by Baseten (REASONING_PROVIDER=baseten); tracking and
voice run on the Meta Model API. No GPU here: fitting, MuJoCo and signal processing are CPU work.

One-time setup (by the developer, never automated):
  pip install modal==1.6.1 && modal token new
  modal secret create physicslens MODEL_API_KEY=... BASETEN_API_KEY=... DEMO_ACCESS_CODE=...
Deploy:
  modal deploy scripts/modal_app.py
then point the site at the printed URL: python scripts/judging_session.py connect --url <url> (or edit backend.json).

Spending stays behind the cost guard and config/budget.yaml (shipped in the image, never in git); Modal compute is
capped by one container, a request timeout and scale-to-zero, and a workspace budget set in Modal's dashboard.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = "/root/app"
SITE = "https://physicslens.vercel.app"
IGNORE = ["**/__pycache__/**", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(str(ROOT / "backend" / "requirements.txt"))
    .add_local_dir(str(ROOT / "backend" / "physicslens"), f"{APP_DIR}/backend/physicslens", ignore=IGNORE)
    .add_local_dir(str(ROOT / "config"), f"{APP_DIR}/config", ignore=IGNORE)
    .add_local_dir(str(ROOT / "gallery"), f"{APP_DIR}/gallery", ignore=IGNORE)
)

app = modal.App("physicslens")
# Runs, the spend ledger and the response cache survive restarts; re-processing a clip stays free.
var = modal.Volume.from_name("physicslens-var", create_if_missing=True)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("physicslens", required_keys=["MODEL_API_KEY", "BASETEN_API_KEY", "DEMO_ACCESS_CODE"])],
    volumes={f"{APP_DIR}/var": var},
    env={"APP_ENV": "demo", "REASONING_PROVIDER": "baseten", "PUBLIC_TRY_ENABLED": "0", "FRONTEND_ORIGINS": SITE,
         "TRUST_PROXY": "1", "PYTHONPATH": f"{APP_DIR}/backend"},
    cpu=2.0,
    memory=4096,
    max_containers=1,  # one container holds every run's state in memory; it also bounds the compute bill
    min_containers=0,  # scale to zero: nothing runs, and nothing is billed, while nobody uses the site
    scaledown_window=600,  # an analysis keeps running in the background while the page polls it
    timeout=1800,
)
@modal.concurrent(max_inputs=40)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, f"{APP_DIR}/backend")
    from physicslens.app import app as fastapi_app

    return fastapi_app
