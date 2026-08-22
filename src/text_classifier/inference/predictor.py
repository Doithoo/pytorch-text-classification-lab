from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..data.tokenizer import SimpleWordTokenizer
from ..models import build_model
from ..training.checkpoint import load_checkpoint
from ..training.train import resolve_device


def predict_text(
    checkpoint_path: str | Path,
    text: str,
    device_name: str = "cpu",
    top_k: int = 3,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = payload["config"]
    labels = payload["label_names"]
    tokenizer_metadata = payload["tokenizer_metadata"]
    tokenizer = SimpleWordTokenizer(
        {str(key): int(value) for key, value in tokenizer_metadata["vocab"].items()},
        int(tokenizer_metadata["max_length"]),
    )
    model = build_model(config["model"]["name"], len(tokenizer.vocab), len(labels), config["model"])
    model.load_state_dict(payload["model_state_dict"])
    device = resolve_device(device_name)
    model.to(device).eval()
    input_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_ids, attention_mask), dim=-1)[0]
    count = min(top_k, len(labels))
    values, indices = probabilities.topk(count)
    predictions = [
        {"label": labels[int(index)], "confidence": float(value)}
        for value, index in zip(values.cpu(), indices.cpu(), strict=True)
    ]
    return {
        "text": text,
        "label": predictions[0]["label"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(labels)},
    }
