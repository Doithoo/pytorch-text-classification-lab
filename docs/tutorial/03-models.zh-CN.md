# 三个模型

[English](03-models.md) | [教程首页](README.zh-CN.md)

三个模型共享输入、标签和评估协议，因此可以在固定 manifest 和随机种子下做受控比较。

| 模型 | 文本表示 | 主要特点 |
| --- | --- | --- |
| `embedding_bag` | mask 加权平均词向量 | 最快、参数少、忽略词序，适合作为基线 |
| `text_cnn` | 多尺寸一维卷积后全局最大池化 | 捕获局部 n-gram，GPU 并行效率高 |
| `bilstm` | 双向 LSTM 最终隐藏状态 | 显式建模顺序，训练通常比 CNN 慢 |

所有模型接收 `[B, L]` 的 `input_ids` 和 `attention_mask`，返回 `[B, C]` logits。EmbeddingBag 用 mask 排除 padding；TextCNN 屏蔽无效卷积窗口并支持极短文本；BiLSTM 按 mask 长度 pack 序列，避免 LSTM 处理尾部 padding。

运行小型模型契约测试：

```bash
uv run python examples/03_minimal_training_loop.py
uv run text-classify list-models
uv run pytest tests/test_models.py
```

模型参数位于 YAML 的 `model` 段。`embedding_dim` 控制词向量宽度；`hidden_dim` 对 TextCNN 表示每个卷积核的 channel 数，对 BiLSTM 表示单方向隐藏维度；`kernel_sizes` 只用于 TextCNN；`num_layers` 和 `bidirectional` 只用于 BiLSTM。

模型选择不能只看训练速度或验证分数。还应记录参数量、显存、吞吐、截断率和高置信度错误。具体建议见[模型选择指南](../guides/choosing-models.zh-CN.md)。
