# 评估与推理

[English](05-evaluation-and-inference.md) | [教程首页](README.zh-CN.md)

模型选择完成后，用 `best.pt` 评估验证或测试 split：

```bash
uv run text-classify evaluate --checkpoint artifacts/tutorial-run/best.pt \
  --manifest-dir data/manifests --split test --device cpu
```

默认输出到 `artifacts/tutorial-run/evaluation/test/`，包含 `metrics.json` 和按置信度降序排列的 `errors.jsonl`。验证集写到独立的 `evaluation/valid/`。已有结果默认不会覆盖；明确需要替换时添加 `--overwrite`。

评估会检查当前 `dataset.json` 的 identity 和标签顺序是否与 checkpoint 一致。这样可以避免在同名但不同划分的数据上静默报告指标。错误样本同时记录数字 ID 和标签名称，便于筛选 `business -> sci_tech` 等混淆。

预测单条文本不需要 manifest：

```bash
uv run text-classify predict --checkpoint artifacts/tutorial-run/best.pt \
  --text "The team won the championship final." --top-k 3 --device cpu
```

输出包含最高标签、置信度、top-k 和所有类别概率。TextCNN 会安全处理空文本和极短文本。也可运行真实示例程序：

```bash
uv run python examples/05_checkpoint_prediction.py \
  --checkpoint artifacts/tutorial-run/best.pt --text "Oil prices rose on Monday."
```

PyTorch checkpoint 使用 pickle。不要加载来源不可信的 `.pt` 文件。单条 softmax 置信度不等于概率校准，也不应被直接解释为业务风险；先结合混淆矩阵和高置信度错误分析。
