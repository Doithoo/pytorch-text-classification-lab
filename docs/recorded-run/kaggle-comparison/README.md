# Kaggle Three-Model Comparison Runner

[中文](README.zh-CN.md) | [Recorded-run index](../README.md)

This directory provides an automated runner with no claimed result yet. It trains EmbeddingBag, TextCNN, and BiLSTM against one AG News manifest, evaluates each test split, and writes `comparison.json` through `compare-runs`.

Before submission, replace the account in `kernel-metadata.json`. Fork users must also update `PROJECT_URL`:

```bash
kaggle kernels push -p docs/recorded-run/kaggle-comparison
```

Expected Kaggle `artifacts/` include three complete run directories, `comparison.json`, and `kaggle-comparison-summary.json`. This page publishes no estimated scores before completion.

When publishing, copy machine-readable evidence except large checkpoints into a new versioned recorded-run directory and state the exact revision. The older TextCNN score cannot be inserted as a comparison row because code and run-metadata protocols differ.
