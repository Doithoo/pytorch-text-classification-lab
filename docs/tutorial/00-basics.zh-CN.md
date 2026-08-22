# 文本分类基础

[English](00-basics.md) | [教程首页](README.zh-CN.md)

文本分类把一段文本映射到一个离散标签。AG News 每个样本由标题、描述和四类标签之一组成：`world`、`sports`、`business`、`sci_tech`。

模型不直接接收字符串。项目先把文本切成 token，再映射到整数 ID。一个 batch 的核心张量是：

```text
input_ids       [batch, sequence]
attention_mask  [batch, sequence]
labels          [batch]
logits          [batch, 4]
```

`logits` 是未归一化分数。训练使用交叉熵比较 logits 和标签 ID；反向传播计算梯度，优化器更新参数。推理时对 logits 使用 softmax 得到概率，但训练损失直接接收 logits。

数据分为训练、验证和测试。训练集更新参数；验证集选择超参数和 `best.pt`；测试集只在选择完成后评估一次。若反复查看测试结果并据此修改模型，测试集就间接参与了训练，最终数字不再是独立估计。

准确率适合 AG News 这种类别均衡任务。macro-F1 会先计算每类 F1 再平均，能更清楚地暴露某一类表现落后。完整定义见[指标参考](../reference/metrics.zh-CN.md)。
