# Scripts

[中文](README.zh-CN.md) | [Documentation index](../docs/README.md)

| Script | Purpose | Network |
| --- | --- | --- |
| `download_data.py` | Download fixed AG News CSV and verify SHA-256 | Yes |
| `generate_doc_assets.py` | Generate curves and confusion from recorded evidence | No |

Download data:

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

An existing file is verified first and mismatch fails. `--force` downloads again but never bypasses hash verification.

Regenerate documentation images with the optional plotting dependency:

```bash
uv run --extra plot python scripts/generate_doc_assets.py \
  --run-dir docs/recorded-run/kaggle-agnews-textcnn --output-dir docs/assets
```

Plots must come from committed machine-readable metrics. Do not edit numbers into images manually.
