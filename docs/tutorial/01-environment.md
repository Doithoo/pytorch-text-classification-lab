# Environment and Installation

[中文](01-environment.zh-CN.md) | [Tutorial index](README.md)

The project supports Python 3.10-3.12 and locks direct and transitive dependencies in `uv.lock`. Start from a fresh clone:

```bash
uv sync --locked --extra dev
uv run text-classify --version
uv run text-classify list-models
uv run pytest
```

These commands do not download data or model weights. `uv run` uses the project environment instead of mixing global Python packages with project dependencies.

Use local CPU for examples, tests, and small runs. `device=auto` selects CUDA, Apple MPS, then CPU. Set `--set device=cpu` when the device must be explicit. Requesting unavailable CUDA or MPS fails immediately.

The full AG News reference run targets a Kaggle T4. Its runner clones the repository and downloads data, so the Kernel must enable Internet and GPU. See the [Kaggle guide](../guides/kaggle.md).

Development dependencies include Ruff, mypy, pytest, build, twine, and pre-commit. Plotting is optional:

```bash
uv run --extra plot python scripts/generate_doc_assets.py \
  --run-dir docs/recorded-run/kaggle-agnews-textcnn
```
