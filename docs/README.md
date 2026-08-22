# Documentation Index

[中文](README.zh-CN.md) | [Project home](../README.md)

Choose a page by goal. Read the tutorial in order; use concepts, guides, and reference pages for design context or exact answers.

## Run the project

1. [Tutorial index](tutorial/README.md): environment through a complete small CPU run.
2. [Kaggle training](guides/kaggle.md): submit a T4 run and download artifacts.
3. [Evaluation and inference](tutorial/05-evaluation-and-inference.md): split outputs, errors, and text prediction.
4. [Troubleshooting](guides/troubleshooting.md): data, devices, sequence length, checkpoints, and outputs.

## Understand the code

- [Classification flow](concepts/classification-flow.md): how text becomes auditable metrics.
- [Code tour](concepts/code-tour.md): package boundaries and reading order.
- [Configuration flow](concepts/configuration-flow.md): defaults, YAML, `--set`, and validation.
- [Model tutorial](tutorial/03-models.md): tensor contracts for EmbeddingBag, TextCNN, and BiLSTM.
- [Architecture decision](architecture/0001-reproducible-text-classification-contracts.md): stable contracts and intentional limits.

## Find an exact answer

| Question | Page |
| --- | --- |
| What does a config field mean? | [Config reference](reference/config-reference.md) |
| What are manifests, labels, and hashes? | [Dataset format](reference/dataset-format.md) |
| How do I read accuracy, macro-F1, and confusion? | [Metrics](reference/metrics.md) |
| What is stored in a checkpoint? | [Checkpoint schema](reference/checkpoint-schema.md) |
| Which model should I use? | [Model catalog](reference/model-zoo.md) |
| What does each CLI command write? | [CLI and outputs](reference/cli-and-outputs.md) |
| How should experiments be compared? | [Experiments](guides/experiments.md) |
| How do I add a model? | [Adding models](guides/adding-models.md) |
| Can AG News data be redistributed? | [Dataset note](reference/ag-news.md) |

Directory-level guides are available for [configs](../configs/README.md), [examples](../examples/README.md), [scripts](../scripts/README.md), and [tests](../tests/README.md).
