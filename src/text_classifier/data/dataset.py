from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

from .manifest import load_manifest
from .tokenizer import SimpleWordTokenizer


class TextClassificationDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str, tokenizer: SimpleWordTokenizer, limit: int | None = None) -> None:
        self.rows = load_manifest(manifest_path, limit)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "id": row["id"],
            "text": row["text"],
            "input_ids": self.tokenizer.encode(row["text"]),
            "label": row["label_id"],
        }


def collate_texts(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_length), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for index, item in enumerate(batch):
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        input_ids[index, : len(ids)] = ids
        attention_mask[index, : len(ids)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


def make_collate_fn(pad_id: int) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result = collate_texts(batch)
        result["input_ids"][result["attention_mask"] == 0] = pad_id
        return result

    return collate
