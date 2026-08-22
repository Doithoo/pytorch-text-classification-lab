# 小型示例

[English](README.md) | [教程](../docs/tutorial/README.zh-CN.md)

示例一次只展示一个概念，前四个使用 CPU，最后一个需要可信 checkpoint。

| 文件 | 概念 | 前置条件 |
| --- | --- | --- |
| `01_tokens.py` | Unicode token、词表和特殊 ID | 无 |
| `02_padding_and_mask.py` | 动态 padding 与 attention mask | 无 |
| `03_minimal_training_loop.py` | logits、交叉熵和反向传播 | 无 |
| `04_prepare_data.py` | 调用配置和 manifest API | 已下载 AG News |
| `05_checkpoint_prediction.py` | 真实加载 checkpoint 并 top-k 推理 | 可信 `best.pt` |

```bash
uv run python examples/01_tokens.py
uv run python examples/02_padding_and_mask.py
uv run python examples/03_minimal_training_loop.py
uv run python examples/04_prepare_data.py
uv run python examples/05_checkpoint_prediction.py \
  --checkpoint artifacts/tutorial-run/best.pt --text "The team won the final."
```

示例不是独立实现；它们调用与 CLI 相同的包 API。完整训练使用 `text-classify train`。
