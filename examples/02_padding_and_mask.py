import torch

from text_classifier.data.dataset import collate_texts

batch = collate_texts(
    [
        {"id": "a", "text": "short", "input_ids": [2, 4, 3], "label": 0},
        {"id": "b", "text": "a longer sentence", "input_ids": [2, 5, 6, 7, 3], "label": 1},
    ]
)
print("input_ids:", batch["input_ids"])
print("attention_mask:", batch["attention_mask"])
print("padding positions:", torch.where(batch["attention_mask"] == 0))
