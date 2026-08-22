# ADR 0001: Reproducible Text-Classification Contracts

[中文](0001-reproducible-text-classification-contracts.zh-CN.md) | [Documentation index](../README.md)

Status: Accepted

## Context

Beginner projects often combine data download, random splitting, vocabulary, training, and test reporting in one notebook. That is easy to execute but makes it difficult to determine whether two results share data, label order, and model-selection protocol.

## Decision

The project uses explicit auditable boundaries: raw data is not committed; preparation writes SHA-256 identified manifests; vocabulary comes only from training; configuration is fully resolved and validated before training; training reads train/valid only; checkpoints contain model, optimizer, tokenizer, labels, and manifest identity; test evaluation writes split-specific non-overwriting evidence; recorded results include environment and errors.

The first scope supports only AG News, a simple word tokenizer, and three classic models. Configuration does not pretend to support unimplemented datasets or Transformers.

## Consequences

Experiments are easier to compare and audit, and tests can enforce CLI and file contracts. The cost is more manifest, metadata, and documentation maintenance. Label, checkpoint, or config schema changes require an explicit compatibility policy.

Future datasets and pretrained models should extend registries and schemas instead of adding implicit branches to the AG News path.
