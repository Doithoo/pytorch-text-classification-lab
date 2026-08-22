# Text-Classification Basics

[中文](00-basics.zh-CN.md) | [Tutorial index](README.md)

Text classification maps a text to a discrete label. Each AG News sample contains a title, description, and one of four labels: `world`, `sports`, `business`, or `sci_tech`.

Models do not receive strings directly. Text is tokenized and mapped to integer IDs. The central batch tensors are:

```text
input_ids       [batch, sequence]
attention_mask  [batch, sequence]
labels          [batch]
logits          [batch, 4]
```

Logits are unnormalized scores. Cross-entropy compares logits with label IDs; backpropagation computes gradients and the optimizer updates parameters. Softmax is used to present probabilities during inference, not before the training loss.

Training data updates parameters, validation data selects settings and `best.pt`, and test data is evaluated once after selection. Repeatedly changing a model after inspecting test results leaks test information into development.

Accuracy is useful for balanced AG News classes. Macro-F1 computes F1 per class before averaging and therefore exposes weak classes more clearly. See the [metrics reference](../reference/metrics.md) for exact definitions.
