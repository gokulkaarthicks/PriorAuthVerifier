.PHONY: setup run dev test lint typecheck check

setup:
	uv sync
	uv run python scripts/download_models.py

run:
	bash scripts/run.sh

dev:
	uv sync
	uv run uvicorn app.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy app

check:
	uv run ruff check .
	uv run mypy app
	uv run pytest
	cd frontend && npm run build
	cd frontend && npm test
