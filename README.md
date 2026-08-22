# PyTorch Text Classification Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-text-classification-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**中文: [README.zh-CN.md](README.zh-CN.md)**

A beginner-oriented PyTorch project that makes text-classification experiments reproducible. AG News is the reference task, while headered generic CSV supports arbitrary binary and multiclass data. The project covers preparation, auditing, stratified manifests, a training-only vocabulary, dynamic padding, classic models, GPU training, evaluation, error analysis, and checkpoint resume.

```text
download -> prepare -> inspect -> dry run -> train -> evaluate -> predict
```

The built-in models are `embedding_bag`, `text_cnn`, and `bilstm`, with `ag_news` and `generic_csv` data adapters. Scope is single-label classic text classification; multilabel data, pretrained Transformers, and production serving remain excluded.

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

This is a bounded recorded run, not a general benchmark. Its config, epoch metrics, environment, confusion matrix, and aggregate test metrics are in the [recorded-run page](docs/recorded-run/kaggle-agnews-textcnn/README.md). Detailed error text is not committed and remains available from Kaggle output.

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

For your own headered `text,label` CSV, start with the [custom-data guide](docs/guides/using-your-data.md) and `configs/generic_csv_example.yaml`. `inspect-data` writes `inspection.json` with duplicates, cross-split leakage, label conflicts, truncation, and OOV. Batch inference and experiment comparison use:

```bash
uv run text-classify predict-file --checkpoint artifacts/first-run/best.pt \
  --input texts.csv --output predictions.jsonl
uv run text-classify compare-runs artifacts/run-a artifacts/run-b
```

A normal run stores the resolved config, tokenizer, `best.pt`, `last.pt`, epoch metrics, and run identity. Training checkpoints use Python pickle and must be trusted; use `export-inference` to create `.safetensors` for distribution.

## Three-model comparison result

The current-code controlled Kaggle run is complete. See the [three-model comparison reference](docs/recorded-run/kaggle-agnews-model-comparison-v0.3.0/README.md) for evidence:

| Model | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| BiLSTM | **0.916053** | **0.915985** |
| EmbeddingBag | 0.915395 | 0.915278 |
| TextCNN | 0.910132 | 0.910090 |

All three rows share the manifest, tokenizer, seed, training budget, and Tesla T4. The older TextCNN record remains as historical revision evidence.

## Kaggle training

Local CUDA is optional. Authenticate the Kaggle CLI and follow the [Kaggle guide](docs/guides/kaggle.md) before submitting:

```bash
uv tool install kaggle
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
```

The runner downloads the repository and AG News, prepares manifests, performs a dry run, trains on CUDA, and evaluates the test split. Download `artifacts/` after completion because Kaggle working storage is temporary.

## Documentation

Use the [documentation index](docs/README.md) to choose a path. Once GitHub Pages is enabled, the same Markdown is published by MkDocs at `https://doithoo.github.io/pytorch-text-classification-lab/`:

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
