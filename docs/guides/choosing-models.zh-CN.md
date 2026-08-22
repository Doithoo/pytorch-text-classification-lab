# 模型选择

[English](choosing-models.md) | [文档首页](../README.zh-CN.md)

先用 `embedding_bag` 建立数据和优化基线，再根据问题选择 TextCNN 或 BiLSTM。

| 场景 | 建议起点 | 原因 |
| --- | --- | --- |
| 验证链路、CPU、小数据 | `embedding_bag` | 快、参数少、错误容易定位 |
| 新闻主题、局部关键词、GPU | `text_cnn` | n-gram 特征强、并行度高，已有实测 |
| 明确关心词序和长程上下文 | `bilstm` | 双向顺序编码，但吞吐较低 |

比较模型时保持 manifest identity、tokenizer 参数、随机种子、epoch、batch size 和优化器一致，只改变模型段和独立 `run_name`。先比较验证集，不应在每次实验后查看测试集。

至少记录最佳验证 macro-F1、训练耗时、参数量、截断率和错误类型。AG News 类别均衡，accuracy 与 macro-F1 接近是合理现象；若两者分离，应先检查逐类支持和混淆矩阵。

当前三模型参考运行显示：BiLSTM test macro-F1 **0.915985**，EmbeddingBag `0.915278`，TextCNN `0.910090`。BiLSTM 分数最高但训练耗时约为 EmbeddingBag 的 3.7 倍；EmbeddingBag 以很低的成本接近最优。TextCNN 的局部 n-gram 设计仍适合教学和 GPU 并行，但本次配置没有超过另外两者。完整协议和限制见[三模型对比参考运行](../recorded-run/kaggle-agnews-model-comparison-v0.3.0/README.zh-CN.md)。
