# Troubleshooting

[中文](troubleshooting.zh-CN.md) | [Documentation index](../README.md)

| Symptom | Cause and action |
| --- | --- |
| Missing `data/manifests/train.csv` | Download data, then run `prepare-data` |
| `unrecognized arguments: --data-dir` | Set data paths through `--config` or `--set data.data_dir=...` |
| Checksum mismatch | Remove the incomplete file and retry; do not bypass verification |
| CUDA requested but unavailable | Use CPU or enable GPU in Kaggle Settings |
| TextCNN kernel exceeds max length | Reduce kernels or increase `data.max_length` |
| Run directory exists | Use a new run name; do not overwrite experiment evidence |
| Evaluation exists | Inspect it; use `--overwrite` only for intentional replacement |
| Manifest identity mismatch | The checkpoint and current split differ; restore the correct manifest |
| Resume config incompatible | Keep model, tokenizer, optimizer, and LR; increase total epochs only |
| Missing `best.pt` | Check epoch completion, non-empty validation, and logs |
| MPS operation fails | Reproduce on CPU and report PyTorch and macOS versions |

Before opening an issue, run:

```bash
uv run text-classify show-config --config <config-path>
uv run ruff check .
uv run pytest
```

Include the exact command, stack trace, OS, Python/PyTorch versions, and minimal config. Never upload private text, Kaggle credentials, or an untrusted checkpoint.
