# Using Your Data

[中文](using-your-data.zh-CN.md) | [Documentation index](../README.md)

The current release officially supports only the three-column AG News CSV and its fixed four labels. Renaming custom files to `train.csv` and `test.csv` is not general compatibility: the parser interprets the first column as 1-4, and label names and prediction follow the AG News protocol.

A temporary no-code experiment must use:

```text
<label 1..4>,<title>,<description>
```

under `<data_dir>/ag_news_csv/`. This is only suitable for structurally compatible four-class experiments. Generated metadata still identifies `ag_news`, so do not publish it as generic custom-data support.

A real dataset addition needs an adapter defining provenance and license, raw schema, label order, stable sample IDs, train/valid/test policy, duplicate handling, hash identity, and class count. Then remove AG News assumptions from training and inference and version the adapter in manifest metadata.

Do not commit private or raw data. Error files retain complete text, so review privacy, copyright, and dataset terms before publication.
