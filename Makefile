# PhysicsLens developer commands (Linux).
# Only `make smoke` spends money, and it asks for confirmation first.

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
export PYTHONPATH := backend

.PHONY: help setup test test-fast smoke report kill unkill dev-backend dev-frontend lint audit numbers site-data

help:
	@echo "setup         create venv and install backend deps"
	@echo "test          run every unit test with mocked models, in parallel (no spend)"
	@echo "test-fast     the same without tests marked slow (over 10 s each)"
	@echo "smoke         ONE real cheap call per model (spends money, asks first)"
	@echo "report        print spend per provider/model/day and remaining budget"
	@echo "kill / unkill block / unblock every paid API call"
	@echo "dev-backend   run FastAPI on :8000"
	@echo "dev-frontend  run Next.js on :3000"
	@echo "lint          ruff over backend, tests and scripts"
	@echo "audit         known vulnerabilities in the pinned Python and npm dependencies"
	@echo "numbers       check every README number against the file it comes from"

setup:
	@test -f config/budget.yaml || cp config/budget.example.yaml config/budget.yaml
	@mkdir -p docs var
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r backend/requirements.txt

test:
	MOCK_MODELS=1 APP_ENV=dev PHYSICSLENS_FIT_WORKERS=1 $(BIN)/pytest -q -n auto tests

test-fast:
	MOCK_MODELS=1 APP_ENV=dev PHYSICSLENS_FIT_WORKERS=1 $(BIN)/pytest -q -n auto -m "not slow" tests

smoke:
	@read -p "This makes REAL paid API calls (capped by smoke_cap_usd in config/budget.yaml). Continue? [y/N] " ans; \
	if [ "$$ans" = "y" ]; then APP_ENV=dev MOCK_MODELS=0 $(BIN)/python -m physicslens.smoke; else echo "Cancelled."; fi

report:
	$(BIN)/python -m physicslens.costguard.report

kill:
	@touch .KILL && echo "Kill switch ON: all paid calls blocked."

unkill:
	@rm -f .KILL && echo "Kill switch OFF."

dev-backend:
	$(BIN)/uvicorn physicslens.app:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

lint:
	$(BIN)/ruff check backend tests scripts

numbers:
	$(BIN)/python scripts/readme_numbers.py

# Everything the static site reads that is generated rather than committed (gallery bundle, sky scenarios, case and
# astronomy data). SKY_FLAGS=--no-discover skips the slow precomputed discoveries.
site-data:
	$(BIN)/python scripts/build_static_gallery.py
	$(BIN)/python scripts/build_astro_static.py
	$(BIN)/python scripts/build_cases.py
	$(BIN)/python scripts/build_sky_static.py $(SKY_FLAGS)

# Known vulnerabilities in the pinned dependencies (queries the public advisory databases; no keys, no spend).
audit:
	$(BIN)/pip install -q pip-audit && $(BIN)/pip-audit -r backend/requirements.txt --progress-spinner off
	cd frontend && npm audit
