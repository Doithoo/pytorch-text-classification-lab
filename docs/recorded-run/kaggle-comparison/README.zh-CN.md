# Kaggle 三模型对比 Runner

[English](README.md) | [参考运行索引](../README.zh-CN.md)

本目录提供当前代码的三模型 runner。它已经在 revision `0b350f2` 完成真实 Tesla T4 运行，发布证据见[三模型对比参考运行](../kaggle-agnews-model-comparison-v0.3.0/README.zh-CN.md)。

runner 会在同一 AG News manifest 上训练 EmbeddingBag、TextCNN 和 BiLSTM，分别评估 test，并用 `compare-runs` 生成 `comparison.json`。未来再次提交时应使用新的版本目录，不覆盖已有结果。

```bash
kaggle kernels push -p docs/recorded-run/kaggle-comparison
```
