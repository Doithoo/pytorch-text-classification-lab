# Using Your Data

[中文](using-your-data.zh-CN.md) | [Documentation index](../README.md)

The `generic_csv` adapter supports arbitrary binary or multiclass tasks. Prepare two UTF-8 CSV files with headers:

```text
data/custom/train.csv
data/custom/test.csv
```

Default columns are `text,label`:

```csv
text,label
The team won the final,sports
Shares rose after earnings,business
```

Each training label needs at least two rows so validation can be sampled per class. Every test label must exist in training. Labels are deduplicated and sorted deterministically, then stored in `dataset.json` and checkpoints.

```bash
uv run text-classify prepare-data --config configs/generic_csv_example.yaml
uv run text-classify inspect-data --config configs/generic_csv_example.yaml
uv run text-classify train --config configs/generic_csv_example.yaml --dry-run --set device=cpu
```

Override different column names:

```bash
uv run text-classify prepare-data --config configs/generic_csv_example.yaml \
  --set data.text_column=body --set data.label_column=category
```

Besides terminal summaries, `inspect-data` writes `inspection.json` with label counts, duplicate IDs, within-split duplicates, cross-split text leakage, conflicting labels, lengths, truncation, vocabulary size, and OOV ratios. Resolve leakage and label conflicts before comparing models.

The adapter currently requires labeled train/test and derives valid from train. It does not support multilabel, unlabeled prediction datasets, hierarchical labels, or streaming data. Do not commit raw private text. Error and batch-prediction files retain text, so review privacy, copyright, and licensing before publication.
