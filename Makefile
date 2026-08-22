PYTHON := uv run python

.PHONY: sync lint format type test check build docs docs-assets

sync:
	uv sync --locked --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

type:
	uv run mypy

test:
	uv run pytest -W error::DeprecationWarning

check: lint type test
	uv run ruff format --check .

build:
	uv build
	uv run twine check dist/*

docs:
	uv run mkdocs build --strict

docs-assets:
	uv run --extra plot $(PYTHON) scripts/generate_doc_assets.py --run-dir docs/recorded-run/kaggle-agnews-textcnn
