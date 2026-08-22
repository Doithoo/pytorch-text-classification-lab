# Training Configurations

[中文](README.zh-CN.md) | [Config reference](../docs/reference/config-reference.md)

Precedence is defaults, YAML, repeated `--set KEY=VALUE`, then centralized validation:

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml
```

| File | Purpose | Data and model |
| --- | --- | --- |
| `learning_minimal.yaml` | CPU dry run and small tutorial | Limited AG News, EmbeddingBag |
| `generic_csv_example.yaml` | Starting point for headered custom CSV | Arbitrary single-label classes, TextCNN |
| `reference_embedding_bag.yaml` | Full Kaggle baseline and comparison | AG News, EmbeddingBag |
| `reference_textcnn.yaml` | Older Kaggle record and current comparison | AG News, TextCNN |
| `reference_bilstm.yaml` | Current Kaggle three-model comparison | AG News, 2-layer BiLSTM |

The older revision has a standalone TextCNN record. All three reference configs completed the same-protocol `0.3.0` comparison; see the [comparison reference run](../docs/recorded-run/kaggle-agnews-model-comparison-v0.3.0/README.md). All embeddings are randomly initialized.

```bash
uv run text-classify prepare-data --config configs/generic_csv_example.yaml
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
```

Give each normal run a unique name. A dry run writes no checkpoint and is not completed training.
