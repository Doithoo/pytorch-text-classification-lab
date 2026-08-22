# CLI 与输出

[English](cli-and-outputs.md) | [文档首页](../README.zh-CN.md)

| 命令 | 作用 | 主要输出 |
| --- | --- | --- |
| `show-config` | 合并并校验配置 | stdout YAML |
| `list-models` | 列出注册模型 | stdout 名称 |
| `prepare-data` | 通过 AG News 或 generic CSV adapter 生成 manifest | CSV、dataset.json、summary.txt |
| `inspect-data` | 长度、标签、重复、泄漏、OOV 审计 | stdout、inspection.json |
| `train --dry-run` | 一个 batch 前向/反向 | stdout，不写运行文件 |
| `train` | 训练和验证 | 运行目录 |
| `evaluate` | 验证或测试 checkpoint | `evaluation/<split>/` |
| `predict` | 单文本 top-k | stdout JSON |
| `predict-file` | CSV/JSONL 批量文本推理 | JSONL |
| `export-inference` | 把可信 `.pt` 转为安全推理权重 | `.safetensors` + JSON sidecar |
| `compare-runs` | 比较相同 manifest 与标签协议的运行 | stdout JSON、可选文件 |

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

评估和批量预测默认拒绝覆盖。`--manifest-dir` 可覆盖 checkpoint 中的机器路径，但 identity 和标签仍必须匹配。批量输入 CSV 要求 `text` 表头；JSONL 每行要求字符串 `text` 字段。输出保留原字段并在 `prediction` 对象中添加预测，明确替换时使用 `--overwrite`。

```bash
uv run text-classify export-inference --checkpoint artifacts/run/best.pt \
  --output artifacts/run/model.safetensors
uv run text-classify predict-file --checkpoint artifacts/run/model.safetensors \
  --input texts.csv --output predictions.jsonl --top-k 3
uv run text-classify compare-runs artifacts/baseline artifacts/textcnn \
  --metric valid_macro_f1 --output artifacts/comparison.json
```

`compare-runs` 在 manifest identity、tokenizer hash 或标签顺序不同时失败，避免静默比较不兼容实验。配置命令统一接受 `--config PATH` 和可重复 `--set KEY=VALUE`。
