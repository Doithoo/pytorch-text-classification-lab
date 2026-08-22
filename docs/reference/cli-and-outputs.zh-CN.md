# CLI 与输出

[English](cli-and-outputs.md) | [文档首页](../README.zh-CN.md)

| 命令 | 作用 | 主要输出 |
| --- | --- | --- |
| `show-config` | 合并并校验配置 | stdout YAML |
| `list-models` | 列出注册模型 | stdout 名称 |
| `prepare-data` | 解析并划分 AG News | manifest CSV、dataset.json、summary.txt |
| `inspect-data` | 长度、截断和标签统计 | stdout |
| `train --dry-run` | 一个 batch 前向/反向 | stdout，不写文件 |
| `train` | 训练和验证 | 运行目录 |
| `evaluate` | 验证或测试 checkpoint | `evaluation/<split>/` |
| `predict` | 单文本 top-k | stdout JSON |

正常运行目录：

```text
artifacts/<run>/
  config.yaml
  tokenizer.json
  metrics.csv
  run.json
  best.pt
  last.pt
  evaluation/
    valid/metrics.json, errors.jsonl
    test/metrics.json, errors.jsonl
```

评估默认拒绝覆盖。`--output` 指定精确目录，`--overwrite` 明确允许替换。`--manifest-dir` 可覆盖 checkpoint 中的机器路径，但 identity 和标签仍必须匹配。

配置命令统一接受 `--config PATH` 和可重复 `--set KEY=VALUE`。数据路径不是独立 `--data-dir` CLI 参数：

```bash
uv run text-classify prepare-data --config configs/learning_minimal.yaml \
  --set data.data_dir=data/raw --set data.manifest_dir=data/manifests
```
