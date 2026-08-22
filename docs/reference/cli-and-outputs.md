# CLI and Outputs

[中文](cli-and-outputs.zh-CN.md) | [Documentation index](../README.md)

| Command | Purpose | Main output |
| --- | --- | --- |
| `show-config` | Merge and validate config | stdout YAML |
| `list-models` | List registered models | stdout names |
| `prepare-data` | Build manifests through AG News or generic CSV adapter | CSV, dataset.json, summary.txt |
| `inspect-data` | Audit lengths, labels, duplicates, leakage, and OOV | stdout, inspection.json |
| `train --dry-run` | One forward/backward batch | stdout, no run files |
| `train` | Train and validate | Run directory |
| `evaluate` | Evaluate validation or test checkpoint | `evaluation/<split>/` |
| `predict` | Single-text top-k | stdout JSON |
| `predict-file` | Batch CSV/JSONL inference | JSONL |
| `export-inference` | Convert trusted `.pt` to safe inference weights | `.safetensors` + JSON sidecar |
| `compare-runs` | Compare runs sharing manifest and label protocols | stdout JSON, optional file |

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

Evaluation and batch prediction refuse overwrite by default. `--manifest-dir` may replace machine-specific checkpoint paths, but identity and labels must match. Batch CSV requires a `text` header; each JSONL row requires a string `text`. Output retains original fields and adds a `prediction` object; use `--overwrite` for intentional replacement.

```bash
uv run text-classify export-inference --checkpoint artifacts/run/best.pt \
  --output artifacts/run/model.safetensors
uv run text-classify predict-file --checkpoint artifacts/run/model.safetensors \
  --input texts.csv --output predictions.jsonl --top-k 3
uv run text-classify compare-runs artifacts/baseline artifacts/textcnn \
  --metric valid_macro_f1 --output artifacts/comparison.json
```

`compare-runs` fails on different manifest identity, tokenizer hash, or label order. Config commands accept `--config PATH` and repeated `--set KEY=VALUE`.
