from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..data.tokenizer import SimpleWordTokenizer
from ..models import build_model
from ..training.checkpoint import load_checkpoint
from ..training.train import resolve_device
from .export import load_inference_checkpoint


class TextPredictor:
    """Reusable predictor backed by one validated checkpoint load."""

    def __init__(self, checkpoint_path: str | Path, device_name: str = "cpu") -> None:
        checkpoint = Path(checkpoint_path)
        payload = (
            load_inference_checkpoint(checkpoint, device="cpu")
            if checkpoint.suffix == ".safetensors"
            else load_checkpoint(checkpoint, map_location="cpu")
        )
        self.labels = [str(label) for label in payload["label_names"]]
        tokenizer_metadata = payload["tokenizer_metadata"]
        self.tokenizer = SimpleWordTokenizer(
            {str(key): int(value) for key, value in tokenizer_metadata["vocab"].items()},
            int(tokenizer_metadata["max_length"]),
        )
        config = payload["config"]
        self.model = build_model(config["model"]["name"], len(self.tokenizer.vocab), len(self.labels), config["model"])
        self.model.load_state_dict(payload["model_state_dict"])
        self.device = resolve_device(device_name)
        self.model.to(self.device).eval()

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        input_ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_ids, attention_mask), dim=-1)[0]
        count = min(top_k, len(self.labels))
        values, indices = probabilities.topk(count)
        predictions = [
            {"label": self.labels[int(index)], "confidence": float(value)}
            for value, index in zip(values.cpu(), indices.cpu(), strict=True)
        ]
        return {
            "text": text,
            "label": predictions[0]["label"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions,
            "probabilities": {label: float(probabilities[index]) for index, label in enumerate(self.labels)},
        }


def predict_text(
    checkpoint_path: str | Path,
    text: str,
    device_name: str = "cpu",
    top_k: int = 3,
) -> dict[str, Any]:
    return TextPredictor(checkpoint_path, device_name).predict(text, top_k)
