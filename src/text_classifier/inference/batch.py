from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .predictor import TextPredictor


def predict_texts(
    checkpoint_path: str | Path,
    texts: Iterable[str],
    device_name: str = "cpu",
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Predict an iterable of texts while reusing one validated checkpoint load."""
    predictor = TextPredictor(checkpoint_path, device_name)
    return [predictor.predict(text, top_k) for text in texts]


def predict_file(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    device_name: str = "cpu",
    top_k: int = 3,
    overwrite: bool = False,
) -> int:
    """Predict a CSV or JSONL file and write one JSON object per input row."""
    source = Path(input_path)
    rows: list[dict[str, Any]] = []
    if source.suffix.lower() == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
                        raise ValueError(f"JSONL row {line_number} must contain a string text field")
                    rows.append(raw)
    else:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "text" not in (reader.fieldnames or []):
                raise ValueError(f"input CSV must contain a text column: {source}")
            rows = [{"text": str(row["text"])} for row in reader]
    predictions = predict_texts(checkpoint_path, (row["text"] for row in rows), device_name, top_k)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"prediction output already exists: {output}; pass overwrite=True to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for original, prediction in zip(rows, predictions, strict=True):
            handle.write(json.dumps({**original, "prediction": prediction}, ensure_ascii=False) + "\n")
    return len(predictions)
