# Tests

[中文](README.zh-CN.md) | [Contributing guide](../CONTRIBUTING.md)

Tests use synthetic CSV, temporary directories, and small models. They download no data or weights. Run all checks with:

```bash
uv run pytest -W error::DeprecationWarning
```

| Test | Main contract |
| --- | --- |
| `test_config.py` | Merge, types, ranges, unknown fields, path safety |
| `test_manifest.py` | Stable stratified split, metadata, invalid ratios |
| `test_data.py` | Training vocabulary, padding, mask |
| `test_models.py` | Three model shapes, short TextCNN input, padding stability |
| `test_checkpoint.py` | Schema, trusted loading, short-text prediction |
| `test_metrics.py` | Confusion, macro metrics, input validation |
| `test_training.py` | Optimizer selection and device failure paths |
| `test_end_to_end.py` | CPU dry run, training, resume, non-overwrite evaluation, CLI prediction |
| `test_documentation.py` | Language pairs, links, commands, configs, examples, model catalog |
| `test_packaging.py` | Build metadata and version entry point |

Test new behavior at the nearest layer and add end-to-end coverage when a cross-module contract changes. Do not depend on network, GPU, the user home directory, or existing `data/`.
