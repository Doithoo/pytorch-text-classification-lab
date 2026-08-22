# 指标参考

[English](metrics.md) | [文档首页](../README.zh-CN.md)

混淆矩阵使用 `confusion[true][predicted]`，行是真实类别、列是预测类别。每类指标为：

```text
precision = TP / predicted positives
recall    = TP / true support
F1        = 2 * precision * recall / (precision + recall)
```

`accuracy` 是全部正确数除以样本数。`macro_precision`、`macro_recall` 和 `macro_f1` 先按类计算再对有 support 的类别等权平均，不按类别样本数加权。空分母按 0 处理。

`metrics.json` 包含 accuracy、四个 macro 数字、每类 precision/recall/F1/support、混淆矩阵和 `label_names`。训练的 `metrics.csv` 只保存每轮训练 loss 与验证集标量，便于画曲线和选择 best。

AG News 测试集四类各 1,900 条，因此 accuracy 与 macro 指标权重接近。已记录运行最大的双向混淆是 business 与 sci_tech：business 被预测为 sci_tech 155 条，反向 129 条。只看总 accuracy 会隐藏这种结构。

指标不衡量概率校准、鲁棒性、公平性或域外泛化。`errors.jsonl` 中的 confidence 是 softmax 最大值，应作为排序线索而不是保证。
