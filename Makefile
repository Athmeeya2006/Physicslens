# PhysicsLens developer commands (Linux).
# Only `make smoke` spends money, and it asks for confirmation first.

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
export PYTHONPATH := backend

.PHONY: help setup test smoke report kill unkill dev-backend dev-frontend lint

help:
	@echo "setup         create venv and install backend deps"
	@echo "test          run unit tests with mocked models (no spend)"
	@echo "smoke         ONE real cheap call per model (spends money, asks first)"
	@echo "report        print spend per provider/model/day and remaining budget"
	@echo "kill / unkill block / unblock every paid API call"
	@echo "dev-backend   run FastAPI on :8000"
	@echo "dev-frontend  run Next.js on :3000"

setup:
	@test -f config/budget.yaml || cp config/budget.example.yaml config/budget.yaml
	@mkdir -p docs var
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r backend/requirements.txt

test:
	MOCK_MODELS=1 APP_ENV=dev $(BIN)/pytest -q tests

smoke:
	@read -p "This makes REAL paid API calls (expected < \$$0.25). Continue? [y/N] " ans; \
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
	$(BIN)/ruff check backend tests
