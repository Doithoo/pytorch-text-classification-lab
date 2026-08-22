# Model Catalog

[中文](model-zoo.zh-CN.md) | [Documentation index](../README.md)

| Registry name | Structure | Mask behavior | Published full result |
| --- | --- | --- | --- |
| `embedding_bag` | Embedding + masked mean + Linear | Excludes padding from mean | No; tests and learning config only |
| `text_cnn` | Embedding + Conv1d 3/4/5 + max + Linear | Masks invalid windows; pads short text | Yes; test macro-F1 0.914610 |
| `bilstm` | Embedding + packed BiLSTM + Linear | Packing skips padding | No; reference config only |

Inspect the runtime registry:

```bash
uv run text-classify list-models
```

`reference_textcnn.yaml` is the measured Kaggle configuration. `reference_bilstm.yaml` is a controlled comparison starting point but has no published full score. `learning_minimal.yaml` uses 256 training rows and EmbeddingBag to validate CPU mechanics, not benchmark performance.

All embeddings are randomly initialized; there are no pretrained word vectors. Parameter count changes substantially with vocabulary and embedding width, so comparisons must retain the same tokenizer.
