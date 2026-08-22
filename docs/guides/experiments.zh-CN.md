# 实验指南

[English](experiments.md) | [文档首页](../README.zh-CN.md)

先写清可证伪问题，例如“相同 manifest、词表和训练预算下，TextCNN 是否比 EmbeddingBag 提高验证 macro-F1”。每次只改变一个主要因素并使用独立 `run_name`。

```bash
uv run text-classify train --config configs/reference_embedding_bag.yaml
uv run text-classify train --config configs/reference_textcnn.yaml
uv run text-classify compare-runs artifacts/kaggle-agnews-embedding-bag \
  artifacts/kaggle-agnews-textcnn --output artifacts/comparison.json
```

`compare-runs` 先检查 `manifest_identity`、`tokenizer_sha256` 和 `label_names`，再从每个 `metrics.csv` 选择指定验证指标的最佳 epoch，并记录 git revision 和耗时。协议不兼容时命令失败，而不是只把最高数字放在一起。

运行前用 `show-config` 确认 seed、数据上限、epoch、batch size、学习率和 optimizer。比较模型时允许模型结构不同，但 tokenizer 和 manifest 应一致。先使用验证指标选择设置，模型选择完成后再分别评估同一个 test manifest。

Kaggle 三模型 runner 位于 `docs/recorded-run/kaggle-comparison/`。它尚未执行，因此没有可发布的新成绩。完成后应保留三个运行目录、comparison、精确 revision 和失败记录，不只发布获胜模型。
