# 配置参考

[English](config-reference.md) | [文档首页](../README.zh-CN.md)

最终配置由默认值、YAML 和 `--set` 合并，并在读取数据前校验。使用 `show-config` 查看真实值。

## data

| 字段 | 默认值 | 约束与含义 |
| --- | --- | --- |
| `name` | `ag_news` | `ag_news` 或带表头的 `generic_csv` |
| `data_dir` | `data/raw` | AG News 根目录，或包含通用 train/test CSV 的目录 |
| `manifest_dir` | `data/manifests` | 准备后 CSV、`dataset.json` 和审计目录 |
| `tokenizer` | `simple_word` | 当前只允许内置词级 tokenizer |
| `text_column` | `text` | generic CSV 正文列名 |
| `label_column` | `label` | generic CSV 标签列名，必须与正文列不同 |
| `vocab_size` | `30000` | 含 4 个特殊 token，至少 4 |
| `min_frequency` | `2` | 训练词表最低词频，正整数 |
| `max_length` | `128` | 含 BOS/EOS，至少 2 |
| `valid_ratio` | `0.1` | 严格位于 0 和 1 之间，按类抽取 |
| `num_workers` | `0` | DataLoader worker，非负整数 |
| `max_*_samples` | `null` | 各 split 调试上限，null 或正整数 |

## model

| 字段 | 默认值 | 约束与含义 |
| --- | --- | --- |
| `name` | `embedding_bag` | `embedding_bag`、`text_cnn`、`bilstm` |
| `embedding_dim` | `128` | 词向量维度，正整数 |
| `hidden_dim` | `128` | CNN channels 或 LSTM 单方向隐藏维度 |
| `dropout` | `0.2` | `[0,1)` |
| `kernel_sizes` | `[3,4,5]` | TextCNN 正整数列表，不超过 max length |
| `num_layers` | `2` | BiLSTM 层数，正整数 |
| `bidirectional` | `true` | BiLSTM 是否双向 |

## train 与运行

| 字段 | 默认值 | 约束与含义 |
| --- | --- | --- |
| `epochs` | `2` | 总 epoch；续训时是目标总数 |
| `batch_size` | `32` | 正整数 |
| `lr` | `0.001` | 正数 |
| `weight_decay` | `0.0001` | 非负数 |
| `optimizer` | `adamw` | `adamw`、`adam`、`sgd` |
| `momentum` | `0.9` | SGD momentum；其他优化器忽略 |
| `seed` | `42` | 非负整数 |
| `amp` | `false` | 只在 CUDA 上启用自动混合精度 |
| `deterministic` | `false` | 请求确定性算法 |
| `grad_clip` | `0.0` | 大于 0 时裁剪梯度范数 |
| `best_metric` | `macro_f1` | accuracy 或四个 macro 指标之一 |
| `device` | `auto` | auto、cpu、mps、cuda、cuda:N |
| `output_dir` | `artifacts` | 运行根目录 |
| `run_name` | `null` | 单个路径组件；null 自动生成时间名 |

命令行示例：

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml \
  --set train.epochs=12 --set train.optimizer=adamw
```
