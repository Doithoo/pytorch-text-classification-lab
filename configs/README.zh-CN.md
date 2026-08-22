# 训练配置

[English](README.md) | [配置字段参考](../docs/reference/config-reference.zh-CN.md)

配置优先级是默认值、YAML、重复的 `--set KEY=VALUE`，最后集中校验：

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml
```

| 文件 | 用途 | 数据规模 | 模型 |
| --- | --- | --- | --- |
| `learning_minimal.yaml` | CPU dry run 和小样本教程 | train 256 / valid 64 / test 64 | EmbeddingBag |
| `reference_textcnn.yaml` | Kaggle 全量实测配置 | 全量 | TextCNN 256d、128 channels |
| `reference_bilstm.yaml` | 同协议序列模型对照起点 | 全量 | 2-layer BiLSTM |

只有 TextCNN 配置有已发布完整成绩。BiLSTM 配置不是性能声明。所有 embedding 随机初始化，不下载预训练权重。

```bash
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
uv run text-classify train --config configs/reference_textcnn.yaml --set device=cuda
```

为每次正常训练设置唯一 `run_name`。dry run 不保存 checkpoint，也不代表训练完成。
