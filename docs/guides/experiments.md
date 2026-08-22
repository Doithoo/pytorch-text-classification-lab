# Experiment Guide

[中文](experiments.zh-CN.md) | [Documentation index](../README.md)

Begin with a falsifiable question such as “under one manifest, vocabulary, and budget, does TextCNN improve validation macro-F1 over EmbeddingBag?” Change one main factor per run and use independent run names.

```bash
uv run text-classify train --config configs/reference_embedding_bag.yaml
uv run text-classify train --config configs/reference_textcnn.yaml
uv run text-classify compare-runs artifacts/kaggle-agnews-embedding-bag \
  artifacts/kaggle-agnews-textcnn --output artifacts/comparison.json
```

`compare-runs` first checks `manifest_identity`, `tokenizer_sha256`, and `label_names`, then selects the best requested validation epoch from each `metrics.csv` and includes revision and elapsed time. It fails on incompatible protocols instead of placing unrelated maxima together.

Use `show-config` to verify seed, sample limits, epochs, batch size, learning rate, and optimizer. Model structure may differ, but tokenizer and manifest should match. Select settings on validation and evaluate the same test manifest only after selection.

The Kaggle three-model runner lives under `docs/recorded-run/kaggle-comparison/`. The `0.3.0` run is complete; see [its evidence](../recorded-run/kaggle-agnews-model-comparison-v0.3.0/README.md). A future completed record should retain all three run directories, comparison, exact revision, and failures, not only the winning model.
