# CLI and Outputs

[中文](cli-and-outputs.zh-CN.md) | [Documentation index](../README.md)

| Command | Purpose | Main output |
| --- | --- | --- |
| `show-config` | Merge and validate configuration | stdout YAML |
| `list-models` | List registered models | stdout names |
| `prepare-data` | Parse and split AG News | manifests, dataset.json, summary.txt |
| `inspect-data` | Length, truncation, label statistics | stdout |
| `train --dry-run` | One forward/backward batch | stdout, no files |
| `train` | Train and validate | Run directory |
| `evaluate` | Evaluate a validation or test split | `evaluation/<split>/` |
| `predict` | Single-text top-k | stdout JSON |

Normal run layout:

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

Evaluation refuses replacement by default. `--output` selects an exact directory and `--overwrite` explicitly replaces it. `--manifest-dir` can replace a machine-specific checkpoint path, but identity and labels must still match.

Configuration commands accept `--config PATH` and repeated `--set KEY=VALUE`. Data paths are not direct `--data-dir` CLI options:

```bash
uv run text-classify prepare-data --config configs/learning_minimal.yaml \
  --set data.data_dir=data/raw --set data.manifest_dir=data/manifests
```
