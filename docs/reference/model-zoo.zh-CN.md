# 模型清单

[English](model-zoo.md) | [文档首页](../README.zh-CN.md)

| 注册名称 | 结构 | mask 行为 | 已发布完整结果 |
| --- | --- | --- | --- |
| `embedding_bag` | Embedding + mask mean + Linear | 均值排除 padding | 是，当前对比 test macro-F1 0.915278 |
| `text_cnn` | Embedding + Conv1d 3/4/5 + max + Linear | 屏蔽无效窗口，短文本安全补齐 | 是，当前对比 test macro-F1 0.910090；旧记录 0.914610 |
| `bilstm` | Embedding + packed BiLSTM + Linear | pack 跳过 padding | 是，当前对比 test macro-F1 **0.915985** |

查看运行时注册表：

```bash
uv run text-classify list-models
```

`reference_textcnn.yaml` 是旧 revision 和当前对比使用的 TextCNN 配置。`reference_embedding_bag.yaml`、`reference_bilstm.yaml` 与 TextCNN 已在 `0.3.0` revision 的同协议运行中完成比较；完整结果见[三模型对比参考运行](../recorded-run/kaggle-agnews-model-comparison-v0.3.0/README.zh-CN.md)。`learning_minimal.yaml` 使用 256 条训练样本，只验证 CPU 链路，不是基准。

所有模型随机初始化 embedding，不使用预训练词向量。参数量随词表和 embedding 维度显著变化；比较时必须保持 tokenizer 相同。
