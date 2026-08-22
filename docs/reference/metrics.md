# Metrics Reference

[中文](metrics.zh-CN.md) | [Documentation index](../README.md)

Confusion uses `confusion[true][predicted]`: rows are true classes and columns are predictions. Per-class metrics are:

```text
precision = TP / predicted positives
recall    = TP / true support
F1        = 2 * precision * recall / (precision + recall)
```

Accuracy is all correct predictions divided by all samples. Macro precision, recall, and F1 compute each supported class first and average classes equally rather than weighting by support. Empty denominators become zero.

`metrics.json` contains accuracy, macro scalars, per-class precision/recall/F1/support, confusion, and `label_names`. Training `metrics.csv` retains only training loss and validation scalars per epoch for curves and best selection.

The AG News test split has 1,900 samples per class, so accuracy and macro weighting are similar. The recorded run's largest two-way confusion is business versus sci_tech: 155 business rows became sci_tech and 129 went the other direction. Aggregate accuracy hides that structure.

These metrics do not measure calibration, robustness, fairness, or out-of-domain generalization. Confidence in `errors.jsonl` is maximum softmax and is a ranking signal, not a guarantee.
