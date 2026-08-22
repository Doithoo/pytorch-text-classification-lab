# Tutorial

[中文](README.zh-CN.md) | [Documentation index](../README.md)

The tutorial follows one AG News workflow. Read and execute it in order:

1. [Text-classification basics](00-basics.md): samples, labels, logits, loss, and splits.
2. [Environment](01-environment.md): uv, CPU checks, and the Kaggle GPU boundary.
3. [Data and tokens](02-data-and-tokens.md): CSV, manifests, vocabulary, truncation, and padding.
4. [Three models](03-models.md): mean pooling, convolutions, and bidirectional LSTM.
5. [Training and resume](04-training.md): dry runs, normal runs, best selection, and last resume.
6. [Evaluation and inference](05-evaluation-and-inference.md): test protocol, errors, and text prediction.

The complete small CPU path is:

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py --data-dir data/raw
uv run text-classify prepare-data --config configs/learning_minimal.yaml
uv run text-classify inspect-data --config configs/learning_minimal.yaml
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
uv run text-classify train --config configs/learning_minimal.yaml --set device=cpu --set run_name=tutorial-run
```

Tutorial metrics only validate the workflow. See the [recorded run](../recorded-run/kaggle-agnews-textcnn/README.md) for the full-data result.
