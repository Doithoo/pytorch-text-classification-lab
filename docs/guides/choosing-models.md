# Choosing a Model

[中文](choosing-models.zh-CN.md) | [Documentation index](../README.md)

Start with `embedding_bag` to establish a data and optimization baseline, then choose TextCNN or BiLSTM from the problem.

| Situation | Starting point | Reason |
| --- | --- | --- |
| Workflow validation, CPU, small data | `embedding_bag` | Fast, small, easy to debug |
| News topics, local keywords, GPU | `text_cnn` | Strong n-gram features, parallel, recorded result |
| Word order and longer context matter | `bilstm` | Bidirectional sequence encoding, lower throughput |

Keep manifest identity, tokenizer settings, seed, epochs, batch size, and optimizer fixed while changing only model settings and run name. Compare on validation first; do not inspect test after every experiment.

Record at least best validation macro-F1, training time, parameter count, truncation, and error types. Accuracy and macro-F1 should be close on balanced AG News. If they diverge, inspect per-class support and confusion.

Only TextCNN has a published full run. The repository does not estimate BiLSTM performance from an unexecuted config. Follow the [experiment guide](experiments.md) to publish a controlled comparison.
