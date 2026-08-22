# Changelog

All notable changes are documented here. The project follows semantic versioning while its public CLI and file contracts stabilize.

## 0.2.0 - 2026-08-22

### Added

- Bilingual tutorial, concept, guide, reference, architecture, and directory documentation.
- Recorded-run curves, a labeled confusion matrix, and an evidence index.
- Central configuration validation with AdamW, Adam, and SGD support.
- Trusted checkpoint validation and reusable single-text inference API.
- Split-specific non-overwriting evaluation outputs and label-aware errors.
- Git/config/tokenizer/lock identities and elapsed time in new run metadata.
- Documentation, checkpoint, metrics, training, packaging, and end-to-end tests.
- Contribution, security, citation, pre-commit, Dependabot, and issue/PR templates.

### Changed

- TextCNN now handles very short input and masks convolution windows beyond valid sequence lengths.
- `inspect-data` reports length percentiles, truncation counts, and label distribution.
- Device resolution checks unavailable CUDA/MPS early and uses CPU by default for evaluation.
- Version metadata is read from installed package metadata.

### Fixed

- Fresh-clone README commands now prepare manifests before dry runs and use supported CLI options.
- The recorded-run index no longer claims that no measured result exists.
- `examples/05_checkpoint_prediction.py` performs real checkpoint inference.
- Invalid validation ratios can no longer create empty or surprising splits.

## 0.1.0 - 2026-08-22

- Initial AG News lab with EmbeddingBag, TextCNN, BiLSTM, Kaggle runner, and first recorded TextCNN result.
