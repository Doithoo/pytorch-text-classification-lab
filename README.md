# PyTorch Text Classification Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-text-classification-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**中文: [README.zh-CN.md](README.zh-CN.md)**

A beginner-oriented PyTorch project that makes text-classification experiments reproducible. AG News is the fixed task used to explain download, stratified manifests, a training-only vocabulary, dynamic padding, classic models, GPU training, evaluation, error analysis, and checkpoint resume.

```text
download -> prepare -> inspect -> dry run -> train -> evaluate -> predict
```

The built-in models are `embedding_bag`, `text_cnn`, and `bilstm`. The current scope deliberately excludes arbitrary datasets, pretrained Transformers, and production serving.

## Recorded Kaggle result

The published TextCNN run trained for eight epochs on a Kaggle Tesla T4. The test split was evaluated once with the checkpoint selected on validation macro-F1.

| Item | Result |
| --- | ---: |
| Train / valid / test | 108,000 / 12,000 / 7,600 |
| Best validation macro-F1 | **0.915909** |
| Test accuracy | **0.914605** |
| Test macro-F1 | **0.914610** |
| Kaggle wall clock | 151.3 seconds |

![AG News TextCNN training curves](docs/assets/ag-news-textcnn-training.png)

![AG News TextCNN test confusion matrix](docs/assets/ag-news-textcnn-confusion.png)

This is a bounded recorded run, not a general benchmark. Its config, tokenizer, epoch metrics, environment, confusion matrix, and 649 errors are in the [recorded-run page](docs/recorded-run/kaggle-agnews-textcnn/README.md).

## Start from a fresh clone

Python 3.10-3.12 and [uv](https://docs.astral.sh/uv/) are required. Run from the repository root:

```bash
git clone https://github.com/Doithoo/pytorch-text-classification-lab.git
cd pytorch-text-classification-lab
uv sync --locked --extra dev
uv run python scripts/download_data.py --data-dir data/raw
uv run text-classify prepare-data --config configs/learning_minimal.yaml
uv run text-classify inspect-data --config configs/learning_minimal.yaml
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
```

The dry run performs one forward, loss, and backward pass without creating artifacts. Continue with a small CPU run:

```bash
uv run text-classify train --config configs/learning_minimal.yaml \
  --set device=cpu --set run_name=first-run
uv run text-classify evaluate --checkpoint artifacts/first-run/best.pt \
  --manifest-dir data/manifests --device cpu
uv run text-classify predict --checkpoint artifacts/first-run/best.pt \
  --text "Stocks rose after the company reported strong earnings." --top-k 3
```

A normal run stores the resolved config, tokenizer, `best.pt`, `last.pt`, epoch metrics, and run identity. PyTorch checkpoints use Python pickle internally; load only files you trust.

## Kaggle training

Local CUDA is optional. Authenticate the Kaggle CLI and follow the [Kaggle guide](docs/guides/kaggle.md) before submitting:

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
```

The runner downloads the repository and AG News, prepares manifests, performs a dry run, trains on CUDA, and evaluates the test split. Download `artifacts/` after completion because Kaggle working storage is temporary.

## Documentation

Use the [documentation index](docs/README.md) to choose a path:

- [Tutorial](docs/tutorial/README.md): basics, environment, data, models, training, evaluation, and inference.
- [Concepts](docs/concepts/classification-flow.md): data flow, code tour, and configuration flow.
- [Guides](docs/guides/choosing-models.md): model choice, experiments, troubleshooting, Kaggle, and extension points.
- [Reference](docs/reference/config-reference.md): config, data, metrics, checkpoints, CLI, and model catalog.
- [Directory guides](configs/README.md): runnable configs, examples, scripts, and tests.

See the [dataset note](docs/reference/ag-news.md) for AG News provenance, citation, and licensing boundaries.

## Development

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
uv run twine check dist/*
```

Behavior changes require tests, and English/Chinese documentation should remain semantically aligned. Read [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the [changelog](CHANGELOG.md) before contributing. Source code is available under the [MIT License](LICENSE).
