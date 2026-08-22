# Data and Tokens

[中文](02-data-and-tokens.zh-CN.md) | [Tutorial index](README.md)

The download script retrieves fixed AG News CSV files and verifies their SHA-256 before installation:

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run text-classify prepare-data --config configs/learning_minimal.yaml
uv run text-classify inspect-data --config configs/learning_minimal.yaml
```

Each AG News source row is `label,title,description`. Preparation joins title and description, converts labels from 1-4 to 0-3, and draws validation rows per class. Custom data uses the `generic_csv` adapter with headered `text,label` files and deterministic labels derived from training. Both adapters emit fixed `id,text,label,label_id` manifests. `dataset.json` records adapter, label order, counts, seed, source hashes, and manifest hashes.

`SimpleWordTokenizer` uses a Unicode regular expression, `casefold()`, and `<pad>`, `<unk>`, `<bos>`, and `<eos>`. Vocabulary counts come only from the training manifest. Long texts preserve BOS/EOS while content is truncated to `max_length`.

Batches are padded dynamically to their longest sample. An attention mask value of one denotes a real token; zero denotes padding. TextCNN also handles sequences shorter than its largest kernel and masks convolution windows outside each sample's valid length.

Run `inspect-data` before training to inspect percentiles, truncation, and class balance. It also writes `inspection.json` with duplicate text, cross-split leakage, label conflicts, and OOV. Increase `max_length` only after considering memory and throughput. See [using your data](../guides/using-your-data.md) and the [dataset format reference](../reference/dataset-format.md).
