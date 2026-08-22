# 文档首页

[English](README.md) | [项目首页](../README.zh-CN.md)

按当前目标选择页面。教程按顺序阅读；概念、指南和参考页用于理解设计或查找具体答案。

## 我想跑通项目

1. [教程首页](tutorial/README.zh-CN.md)：从环境到一次完整 CPU 小样本运行。
2. [Kaggle 训练](guides/kaggle.zh-CN.md)：提交 T4 GPU 训练并下载产物。
3. [评估与推理](tutorial/05-evaluation-and-inference.zh-CN.md)：评估目录、错误样本和单文本预测。
4. [排错](guides/troubleshooting.zh-CN.md)：数据、设备、卷积长度、checkpoint 和输出问题。

## 我想理解代码

- [文本分类流程](concepts/classification-flow.zh-CN.md)：一段文本如何变成可审计指标。
- [代码导览](concepts/code-tour.zh-CN.md)：包边界和建议阅读顺序。
- [配置流](concepts/configuration-flow.zh-CN.md)：默认值、YAML、`--set` 与验证。
- [模型教程](tutorial/03-models.zh-CN.md)：EmbeddingBag、TextCNN 和 BiLSTM 的张量契约。
- [架构决策](architecture/0001-reproducible-text-classification-contracts.zh-CN.md)：稳定契约和有意限制。

## 我需要具体答案

| 问题 | 页面 |
| --- | --- |
| 配置字段是什么意思？ | [配置参考](reference/config-reference.zh-CN.md) |
| manifest、标签和哈希是什么？ | [数据格式](reference/dataset-format.zh-CN.md) |
| accuracy、macro-F1 和混淆矩阵怎么读？ | [指标参考](reference/metrics.zh-CN.md) |
| checkpoint 中保存什么？ | [checkpoint 协议](reference/checkpoint-schema.zh-CN.md) |
| 应该选择哪个模型？ | [模型清单](reference/model-zoo.zh-CN.md) |
| CLI 会写哪些文件？ | [CLI 与输出](reference/cli-and-outputs.zh-CN.md) |
| 如何设计可比较实验？ | [实验指南](guides/experiments.zh-CN.md) |
| 如何添加模型？ | [添加模型](guides/adding-models.zh-CN.md) |
| AG News 能否再分发？ | [数据集说明](reference/ag-news.zh-CN.md) |

配置、示例、脚本和测试目录也有独立入口：[configs](../configs/README.zh-CN.md)、[examples](../examples/README.zh-CN.md)、[scripts](../scripts/README.zh-CN.md)、[tests](../tests/README.zh-CN.md)。
