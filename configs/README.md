# Training Configurations

[中文](README.zh-CN.md) | [Config reference](../docs/reference/config-reference.md)

Precedence is defaults, YAML, repeated `--set KEY=VALUE`, then centralized validation:

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml
```

| File | Purpose | Data scale | Model |
| --- | --- | --- | --- |
| `learning_minimal.yaml` | CPU dry run and small tutorial | train 256 / valid 64 / test 64 | EmbeddingBag |
| `reference_textcnn.yaml` | Measured full Kaggle config | Full | TextCNN 256d, 128 channels |
| `reference_bilstm.yaml` | Sequence-model comparison starting point | Full | 2-layer BiLSTM |

Only TextCNN has a published full score. The BiLSTM config is not a performance claim. All embeddings are randomly initialized and download no pretrained weights.

```bash
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
uv run text-classify train --config configs/reference_textcnn.yaml --set device=cuda
```

Give each normal run a unique name. A dry run writes no checkpoint and is not completed training.
