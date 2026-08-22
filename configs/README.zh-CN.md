# 训练配置

[English](README.md) | [配置字段参考](../docs/reference/config-reference.zh-CN.md)

配置优先级是默认值、YAML、重复的 `--set KEY=VALUE`，最后集中校验：

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml
```

| 文件 | 用途 | 数据与模型 |
| --- | --- | --- |
| `learning_minimal.yaml` | CPU dry run 和小样本教程 | AG News 上限样本、EmbeddingBag |
| `generic_csv_example.yaml` | 自有带表头 CSV 起点 | 任意单标签类别、TextCNN |
| `reference_embedding_bag.yaml` | Kaggle 全量快速基线 | AG News、EmbeddingBag |
| `reference_textcnn.yaml` | Kaggle 已有旧实测与新对比配置 | AG News、TextCNN |
| `reference_bilstm.yaml` | Kaggle 序列模型对照 | AG News、2-layer BiLSTM |

只有旧 revision 的 TextCNN 有已发布完整成绩。三个 reference 配置的新协议对比尚待 Kaggle runner 完成，不能把配置当性能声明。所有 embedding 随机初始化。

```bash
uv run text-classify prepare-data --config configs/generic_csv_example.yaml
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
```

为每次正常训练设置唯一 `run_name`。dry run 不保存 checkpoint，也不代表训练完成。
