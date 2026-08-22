from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def classification_metrics(labels: list[int], predictions: list[int], num_classes: int) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if any(value < 0 or value >= num_classes for value in [*labels, *predictions]):
        raise ValueError("labels and predictions must be valid class IDs")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        matrix[label, prediction] += 1
    support = matrix.sum(axis=1)
    tp = np.diag(matrix).astype(float)
    precision = tp / np.maximum(matrix.sum(axis=0), 1.0)
    recall = tp / np.maximum(support, 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    supported = support > 0
    return {
        "accuracy": float(tp.sum() / max(matrix.sum(), 1)),
        "macro_f1": float(f1[supported].mean()) if supported.any() else 0.0,
        "macro_precision": float(precision[supported].mean()) if supported.any() else 0.0,
        "macro_recall": float(recall[supported].mean()) if supported.any() else 0.0,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.astype(int).tolist(),
        "confusion": matrix.tolist(),
    }


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
