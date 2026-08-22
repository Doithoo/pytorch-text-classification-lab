# Adding a Model

[中文](adding-models.zh-CN.md) | [Documentation index](../README.md)

A built-in model follows one contract: construction receives vocabulary and class counts; `forward(input_ids, attention_mask)` receives two `[B,L]` integer tensors and returns `[B,C]` logits. Do not apply softmax inside forward.

A model addition must update:

1. Implement the module under `src/text_classifier/models/` and handle padding.
2. Register a stable name in `build_model()` and expose it through `list_models()`.
3. Declare allowed configuration fields and ranges in validation.
4. Test shape, short input, padding, backward, and checkpoint round-trip.
5. Update the model catalog, config reference, example config, and both languages.

If a model needs a different tokenizer or batch contract, do not spread conditionals through existing classifiers. Propose the new data/model contract and checkpoint compatibility policy first.

Pretrained Transformers additionally require model/tokenizer revisions, cache policy, attention/token-type fields, weight licensing, differential learning rates, and safe serialization. Those capabilities are not implemented; adding a dependency and name alone is not support.
