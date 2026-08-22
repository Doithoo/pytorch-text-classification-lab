# Model Catalog

[中文](model-zoo.zh-CN.md) | [Documentation index](../README.md)

| Registry name | Structure | Mask behavior | Published full result |
| --- | --- | --- | --- |
| `embedding_bag` | Embedding + masked mean + Linear | Excludes padding from mean | Yes; current comparison test macro-F1 0.915278 |
| `text_cnn` | Embedding + Conv1d 3/4/5 + max + Linear | Masks invalid windows; pads short text | Yes; current comparison 0.910090; older record 0.914610 |
| `bilstm` | Embedding + packed BiLSTM + Linear | Packing skips padding | Yes; current comparison test macro-F1 **0.915985** |

Inspect the runtime registry:

```bash
uv run text-classify list-models
```

`reference_textcnn.yaml` is used by the older record and current comparison. `reference_embedding_bag.yaml`, `reference_bilstm.yaml`, and TextCNN completed the same-protocol `0.3.0` comparison; see the [three-model comparison reference run](../recorded-run/kaggle-agnews-model-comparison-v0.3.0/README.md). `learning_minimal.yaml` uses 256 training rows to validate CPU mechanics, not benchmark performance.

All embeddings are randomly initialized; there are no pretrained word vectors. Parameter count changes substantially with vocabulary and embedding width, so comparisons must retain the same tokenizer.
