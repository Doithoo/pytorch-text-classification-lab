# 文本分类流程

[English](classification-flow.md) | [文档首页](../README.zh-CN.md)

一条样本的主路径是：

```text
AG News CSV
  -> prepare_data: 固定 id、标签顺序、分层 valid、SHA-256
  -> manifest CSV
  -> training-only vocabulary
  -> TextClassificationDataset: text -> token IDs
  -> collate: 动态 padding + attention mask
  -> classifier: [B,L] -> logits [B,C]
  -> cross entropy / optimizer
  -> validation metrics -> best.pt
  -> one-shot test metrics + ranked errors
```

身份链路与张量链路同样重要。`dataset.json` 标识源和划分；`tokenizer.json` 标识词表；`config.yaml` 标识实际超参数；checkpoint 把它们关联；`run.json` 记录代码、环境和哈希。缺少其中任何一个，都可能得到一个可加载但无法解释来源的模型。

训练只读取 train 和 valid。test 由显式 `evaluate` 命令读取。这个边界使“选择模型”和“报告最终结果”在代码路径上分离，而不只依赖使用者自律。

错误分析不是训练的附属输出。`errors.jsonl` 按错误预测置信度排序，适合先检查模型最确信但错误的样本，再按真实/预测标签对归类问题。
