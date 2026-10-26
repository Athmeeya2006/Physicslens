"""FastAPI entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from .costguard.guard import Guard
from .costguard.ledger import Ledger
from .settings import load_budget, load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Raises before the server accepts a request if MOCK_MODELS=1 in demo/prod or budget.yaml is missing.
settings = load_settings()
_guard = Guard(Ledger(settings.ledger_path), load_budget(settings.config_dir))

app = FastAPI(title="PhysicsLens")


@app.get("/api/health")
def health() -> dict[str, object]:
    # mock_mode drives the red "MOCK MODE" banner in the UI.
    return {"status": "ok", "app_env": settings.app_env, "mock_mode": settings.mock_models}


@app.get("/api/budget")
def budget() -> dict[str, object]:
    # Dev UI header only: judges and public visitors must never see quota numbers.
    if settings.app_env != "dev":
        raise HTTPException(status_code=404)
    return _guard.summary()
