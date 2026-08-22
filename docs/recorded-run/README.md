# Recorded-Run Evidence

[中文](README.zh-CN.md) | [Documentation index](../README.md)

This directory contains the reproducible Kaggle runner, Kernel metadata, and machine-readable evidence from completed runs.

Published run:

| Run | Model | Test accuracy | Test macro-F1 | Page |
| --- | --- | ---: | ---: | --- |
| `kaggle-agnews-textcnn` | TextCNN | 0.914605 | 0.914610 | [Details](kaggle-agnews-textcnn/README.md) |

A run retains exact git revision, resolved config, source and manifest identity, tokenizer, epoch metrics, Kaggle GPU/PyTorch/CUDA/Python, test metrics, and high-confidence errors. The 93 MB checkpoints remain in Kaggle output instead of Git.

`kaggle/` is the resubmittable single-model runner and `kaggle-comparison/` is a three-model runner with no published result yet; neither is completed-run evidence. Add a separate directory for new results and do not replace historical evidence. Never describe dry-run, sample-limited, or pending configurations as a full benchmark.
