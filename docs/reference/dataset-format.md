# Dataset Format and Identity

[中文](dataset-format.zh-CN.md) | [Documentation index](../README.md)

Raw AG News files live at:

```text
<data_dir>/ag_news_csv/train.csv
<data_dir>/ag_news_csv/test.csv
```

A CSV parser reads every row as `label,title,description`. Quotes, commas, and newlines must follow standard CSV; do not split lines manually.

All three prepared manifests share this schema:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable source-row ID such as `train-000000` |
| `text` | string | Joined title and description |
| `label` | string | world, sports, business, or sci_tech |
| `label_id` | integer | 0-3 in that order |

`dataset.json` has `schema_version=1` and stores dataset name, label order, seed, validation ratio, split counts, source SHA-256, and manifest SHA-256. Manifest identity is SHA-256 of this metadata serialized with sorted keys. Checkpoints retain it and resume/evaluation compare it.

Preparation shuffles and samples validation independently per class, then restores stable ID order. Equal sources, seed, and ratio produce equal manifests. Any source, split, or label-protocol change should produce a new identity.
