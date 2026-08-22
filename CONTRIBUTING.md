# Contributing Guide

[中文](CONTRIBUTING.zh-CN.md)

## Environment

```bash
uv sync --locked --extra dev
uv run pre-commit install
```

Work on a focused branch. Behavior changes require tests, and public commands, configuration, or file-contract changes require matching English and Chinese documentation. Do not commit raw data, checkpoints, Kaggle credentials, private text, or unrelated generated files.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv run pre-commit run --all-files
uv build
uv run twine check dist/*
```

Tests must not download AG News, require GPU, read the user home directory, or load external checkpoints. Documentation commands should run from the repository root. See `src/README.md`, `scripts/README.md`, and `tests/README.md` for ownership boundaries.

## Data and experiments

Data or experiment changes must record source URLs and checksums, manifest identity, label order, resolved config, dependency lock, exact revision, and command. Select models on validation and evaluate final test once. The source-code MIT License does not replace dataset licensing and privacy obligations.

## Commits and pull requests

Use English ASCII [Conventional Commits](https://www.conventionalcommits.org/), for example `fix(models): Handle short TextCNN inputs`. Keep the subject concise and omit a trailing period.

A PR should explain the problem, behavior change, test evidence, compatibility impact, and documentation. Incompatible checkpoint, manifest, config, or CLI changes require a migration plan.
