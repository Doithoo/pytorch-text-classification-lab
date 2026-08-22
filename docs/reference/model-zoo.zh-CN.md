# 模型清单

[English](model-zoo.md) | [文档首页](../README.zh-CN.md)

| 注册名称 | 结构 | mask 行为 | 已发布完整结果 |
| --- | --- | --- | --- |
| `embedding_bag` | Embedding + mask mean + Linear | 均值排除 padding | 否，仅测试与学习配置 |
| `text_cnn` | Embedding + Conv1d 3/4/5 + max + Linear | 屏蔽无效窗口，短文本安全补齐 | 是，测试 macro-F1 0.914610 |
| `bilstm` | Embedding + packed BiLSTM + Linear | pack 跳过 padding | 否，仅参考配置 |

查看运行时注册表：

```bash
uv run text-classify list-models
```

`reference_textcnn.yaml` 是 Kaggle 实测配置。`reference_bilstm.yaml` 是受控对照起点，但没有发布完整成绩。`learning_minimal.yaml` 使用 256 条训练样本和 EmbeddingBag，只验证 CPU 链路，不是基准。

所有模型随机初始化 embedding，不使用预训练词向量。参数量随词表和 embedding 维度显著变化；比较时必须保持 tokenizer 相同。
