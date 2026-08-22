# Source Layout

[中文](README.zh-CN.md) | [Code tour](../docs/concepts/code-tour.md)

The installable package is `text_classifier`:

```text
config.py               Configuration merge and validation
cli.py                  text-classify entry point
data/                    AG News/generic CSV adapters, manifests, audit, tokenizer, Dataset
models/                  Three classifiers and registry
training/                Training loop, checkpoints, run metadata
evaluation/              Classification metrics and compatible-run comparison
inference/               Trusted-checkpoint text and file prediction
```

Dependency direction is CLI -> data/model/training/evaluation/inference; lower modules should not import CLI. Training and inference share model, tokenizer, and checkpoint contracts instead of duplicating deserialization.

The public Python API remains small and the CLI is the primary interface. Renaming registry names, config fields, manifest columns, or checkpoint fields is a compatibility change that needs tests and migration documentation.
