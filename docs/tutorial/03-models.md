# The Three Models

[中文](03-models.zh-CN.md) | [Tutorial index](README.md)

All three models share inputs, labels, and evaluation protocol, so they can be compared under a fixed manifest and seed.

| Model | Representation | Main property |
| --- | --- | --- |
| `embedding_bag` | Masked mean of token embeddings | Fast, small, ignores word order, useful baseline |
| `text_cnn` | Multi-width 1-D convolution and global max pooling | Captures local n-grams and parallelizes well |
| `bilstm` | Final hidden state of a bidirectional LSTM | Models order explicitly and is usually slower |

Each model receives `[B, L]` input IDs and masks and returns `[B, C]` logits. EmbeddingBag excludes padding from its mean. TextCNN masks invalid convolution windows and accepts very short text. BiLSTM packs sequences from mask lengths so its recurrent encoder skips trailing padding.

Run the small model contract examples and tests:

```bash
uv run python examples/03_minimal_training_loop.py
uv run text-classify list-models
uv run pytest tests/test_models.py
```

Model parameters live under the YAML `model` section. `embedding_dim` is the token vector width. `hidden_dim` is channels per TextCNN kernel or hidden units per BiLSTM direction. `kernel_sizes` applies only to TextCNN; `num_layers` and `bidirectional` apply only to BiLSTM.

Do not select a model from validation score alone. Record parameter count, memory, throughput, truncation, and high-confidence errors. See [choosing models](../guides/choosing-models.md).
