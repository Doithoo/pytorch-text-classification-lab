# Kaggle Three-Model Comparison Runner

[中文](README.zh-CN.md) | [Recorded-run index](../README.md)

This directory provides the current-code three-model runner. It completed a real Tesla T4 run at revision `0b350f2`; see the [comparison reference run](../kaggle-agnews-model-comparison-v0.3.0/README.md) for published evidence.

The runner trains EmbeddingBag, TextCNN, and BiLSTM on one AG News manifest, evaluates each test split, and writes `comparison.json` through `compare-runs`. Future submissions should use a new versioned directory and never replace an existing result.

```bash
kaggle kernels push -p docs/recorded-run/kaggle-comparison
```
