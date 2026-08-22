# Checkpoint Schema

[中文](checkpoint-schema.zh-CN.md) | [Documentation index](../README.md)

A project checkpoint is a mapping written by `torch.save`; current `schema_version=1`. Required content includes:

| Field | Content |
| --- | --- |
| `model_name`, `model_config` | Registry name and construction settings |
| `model_state_dict` | Model parameters |
| `optimizer_state_dict` | Normal resume state |
| `scaler_state_dict` | AMP GradScaler state, possibly null |
| `epoch`, `metrics` | Completed epoch index and validation metrics |
| `config` | Fully resolved run configuration |
| `tokenizer_metadata` | Vocabulary, max length, and tokenizer name |
| `manifest_identity` | Training-data protocol identity |
| `label_names` | Class-ID order; old v1 files fall back to AG News labels |

The loader validates file existence, mapping type, schema, required fields, configuration, and basic tokenizer structure. It fills current defaults when an older v1 config lacks later additive fields.

Use `best.pt` for evaluation and inference and `last.pt` for interrupted training. Do not retain weights while discarding config, tokenizer, and manifest metadata.

`torch.load(weights_only=False)` uses pickle, and a malicious file can execute code. Load only checkpoints produced by you or a trusted project run. Schema validation occurs after deserialization and cannot prevent pickle execution.
