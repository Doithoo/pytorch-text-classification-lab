# ADR 0002: Generic CSV Dataset Contract

[中文](0002-generic-csv-dataset-contract.zh-CN.md) | [Documentation index](../README.md)

Status: Accepted

## Context

In `0.2.0`, models and checkpoints used metadata labels, but preparation still hardcoded AG News paths, headerless columns, and four labels. Dynamic class count therefore did not provide real custom-data support.

## Decision

Retain the `ag_news` adapter for recorded-run compatibility and add `generic_csv`. It reads headered train/test CSV, accepts configurable column names, derives deterministic IDs from sorted training labels, and samples validation per class. Both adapters emit one manifest schema and identity protocol.

Auditing is separate evidence after preparation: duplicate IDs, repeated text, cross-split leakage, conflicting labels, lengths, truncation, and tokenizer OOV. Training consumes manifests and remains unaware of the raw adapter.

## Consequences

Arbitrary binary and multiclass data can reuse all three models, checkpoints, evaluation, and inference. Label sorting becomes a public contract; changing columns, label text, or adapter changes identity. Multilabel, unlabeled test, and streaming input require future schemas and cannot be hidden in this protocol.
