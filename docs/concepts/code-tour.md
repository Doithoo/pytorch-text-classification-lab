# Code Tour

[中文](code-tour.zh-CN.md) | [Documentation index](../README.md)

Read in data-flow order rather than alphabetical directory order:

1. `src/text_classifier/config.py`: defaults, YAML merge, CLI overrides, validation.
2. `data/manifest.py`: AG News parsing, stratified split, and dataset identity.
3. `data/tokenizer.py` and `data/dataset.py`: token IDs, truncation, padding, masks.
4. `models/classifiers.py`: three `[B,L] -> [B,C]` models.
5. `training/train.py`: device, optimizer, epochs, validation, checkpoints, metadata.
6. `training/checkpoint.py`: trusted checkpoint validation and old-default normalization.
7. `inference/predictor.py` and `evaluation/metrics.py`: prediction and metrics.
8. `cli.py`: user-facing orchestration of those modules.

The CLI remains thin: parsing, path policy, and JSON presentation belong there; training, inference, and preparation remain testable Python APIs. Preserve that boundary when adding behavior.

`examples/` demonstrates one concept per program and should not duplicate the trainer. `scripts/` contains repository tasks such as downloads and documentation plots. `docs/recorded-run/` is published evidence, not package runtime data.
