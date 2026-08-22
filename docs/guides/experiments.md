# Experiment Guide

[中文](experiments.zh-CN.md) | [Documentation index](../README.md)

Begin with a question such as “under the same vocabulary and training budget, does TextCNN improve validation macro-F1 over EmbeddingBag?” Change one main factor per run and use unique names:

```bash
uv run text-classify train --config configs/learning_minimal.yaml \
  --set run_name=baseline --set model.name=embedding_bag
uv run text-classify train --config configs/learning_minimal.yaml \
  --set run_name=textcnn --set model.name=text_cnn
```

Save `show-config` output first and verify manifest, seed, sample limits, epochs, batch size, learning rate, and optimizer. The commands above demonstrate mechanics; a formal comparison needs sufficient data and valid kernel/max-length settings.

Compare best validation values in `metrics.csv` and time, revision, config/tokenizer/manifest hashes in `run.json`. Evaluate the same test manifest only after model selection. Report negative and failed runs instead of retaining only the highest score.

There is no automatic aggregation command yet because incompatible protocols should not be merged silently. Use an explicit table today; a future `compare-runs` must validate identities and metric schema first.
