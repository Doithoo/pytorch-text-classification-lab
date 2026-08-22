# Kaggle 三模型对比 Runner

[English](README.md) | [参考运行索引](../README.zh-CN.md)

本目录提供尚未声明成绩的自动化 runner。它在同一 AG News manifest 上依次训练 EmbeddingBag、TextCNN 和 BiLSTM，分别评估 test，并用 `compare-runs` 生成 `comparison.json`。

提交前修改 `kernel-metadata.json` 的用户名；fork 用户还要修改 runner 的 `PROJECT_URL`：

```bash
kaggle kernels push -p docs/recorded-run/kaggle-comparison
```

预期产物位于 Kaggle `artifacts/`，包括三个完整运行目录、`comparison.json` 和 `kaggle-comparison-summary.json`。runner 完成之前，本目录不发布任何估算成绩。

结果发布时应复制除大型 checkpoint 外的机器证据到新的、带版本的 recorded-run 目录，并记录精确 git revision。当前旧 TextCNN 结果不能作为这个 runner 的对比行，因为代码和运行 metadata 协议不同。
