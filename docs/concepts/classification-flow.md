# Text-Classification Flow

[中文](classification-flow.zh-CN.md) | [Documentation index](../README.md)

A sample follows this path:

```text
AG News CSV
  -> prepare_data: stable IDs, label order, stratified valid, SHA-256
  -> manifest CSV
  -> training-only vocabulary
  -> TextClassificationDataset: text -> token IDs
  -> collate: dynamic padding + attention mask
  -> classifier: [B,L] -> logits [B,C]
  -> cross entropy / optimizer
  -> validation metrics -> best.pt
  -> one-shot test metrics + ranked errors
```

Identity flow matters as much as tensor flow. `dataset.json` identifies source and splits, `tokenizer.json` identifies the vocabulary, `config.yaml` identifies resolved settings, the checkpoint joins them, and `run.json` records code, environment, and hashes. A model missing one of these may load but cannot be interpreted reliably.

Training reads only train and valid. Test is read by the explicit `evaluate` command. This separates model selection from final reporting in code rather than relying only on user discipline.

Error analysis is a first-class output. `errors.jsonl` sorts mistakes by predicted confidence so review can begin with confidently wrong examples and then group them by true/predicted label pairs.
