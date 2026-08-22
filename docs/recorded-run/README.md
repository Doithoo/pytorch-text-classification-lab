# Recorded-Run Index

[中文](README.zh-CN.md) | [Documentation index](../README.md)

This directory contains reproducible Kaggle runners, Kernel metadata, and machine-readable evidence from completed runs. Large checkpoints stay out of Git and are downloadable from each run's Kaggle page.

## Published runs

| Run | Model | Test accuracy | Test macro-F1 | Page |
| --- | --- | ---: | ---: | --- |
| `kaggle-agnews-textcnn` | TextCNN, older revision | 0.914605 | 0.914610 | [Details](kaggle-agnews-textcnn/README.md) |
| `kaggle-agnews-model-comparison-v0.3.0` | EmbeddingBag / TextCNN / BiLSTM | 0.915395 / 0.910132 / 0.916053 | 0.915278 / 0.910090 / **0.915985** | [Details](kaggle-agnews-model-comparison-v0.3.0/README.md) |

## Runners

`kaggle/` is the single-model runner and `kaggle-comparison/` is the current three-model runner. The latter completed a real T4 run at revision `0b350f2`; future experiments should use new versioned directories and never replace historical evidence.

Every complete run should retain exact git revision, source and manifest identity, tokenizer, resolved config, epoch metrics, GPU/PyTorch/CUDA/Python, test metrics, errors, and summary. Select with validation only, then evaluate test once.
