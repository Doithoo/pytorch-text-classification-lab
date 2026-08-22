# PyTorch Text Classification Lab

A beginner-oriented, reproducible PyTorch project for learning text classification on Kaggle GPU.

The first reference task is AG News, with a complete path from raw CSV files to token IDs, padded batches,
training, evaluation, error analysis, and a checkpoint that can be resumed. The primary training environment is
Kaggle with a T4 GPU. Local CPU execution is kept for tests and short dry runs.

## Kaggle first

The recommended path is documented in [the Kaggle guide](docs/guides/kaggle.zh-CN.md). The short version is:

```bash
kaggle auth login
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <your-user>/pytorch-text-classification-lab-agnews-gpu
kaggle kernels output <your-user>/pytorch-text-classification-lab-agnews-gpu --file-pattern 'artifacts/.*' -p kaggle-output
```

The runner enables Internet, downloads AG News into the writable `/kaggle/working` volume, prepares fixed
manifests, runs a dry run, trains on CUDA, evaluates the test split, and leaves all evidence under `artifacts/`.

## Local development

```bash
uv sync --locked --extra dev
uv run text-classify show-config --config configs/learning_minimal.yaml
uv run pytest
uv run text-classify train --config configs/learning_minimal.yaml --dry-run
```

Local data preparation is also supported when `data/raw/ag_news_csv/{train,test}.csv` is present:

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run text-classify prepare-data --data-dir data/raw --manifest-dir data/manifests
```

## Learning route

1. `examples/01_tokens.py`: Unicode text and the vocabulary.
2. `examples/02_padding_and_mask.py`: variable-length batches and attention masks.
3. `examples/03_minimal_training_loop.py`: logits, loss, and one optimizer step.
4. `docs/tutorial/`: the complete data, model, training, and evaluation route.
5. Compare `embedding_bag`, `text_cnn`, and `bilstm` using the same manifest and seed.

## Project structure

```text
configs/                  Small and reference experiment configurations
scripts/                  Data download and inspection utilities
docs/                     Tutorials, Kaggle workflow, and recorded evidence
examples/                 Small executable learning programs
src/text_classifier/      Installed application and reusable code
tests/                    Unit, integration, and CLI tests
```

The raw dataset is not committed. Check its source and license terms before redistribution. Training only builds
the vocabulary from the training split, and every run records the manifest and tokenizer identities.

## Development checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
uv run twine check dist/*
```

MIT License.
