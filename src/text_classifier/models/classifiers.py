from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class EmbeddingBagClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.dropout(pooled))


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        kernel_sizes: list[int] | None = None,
        channels: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        sizes = kernel_sizes or [3, 4, 5]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, size) for size in sizes])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * len(sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(features, dim=1)))


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).cpu()
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.encoder(packed)
        features = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.encoder.bidirectional else hidden[-1]
        return self.classifier(self.dropout(features))


def build_model(name: str, vocab_size: int, num_classes: int, params: Mapping[str, Any]) -> nn.Module:
    embedding_dim = int(params.get("embedding_dim", 128))
    dropout = float(params.get("dropout", 0.2))
    if name == "embedding_bag":
        return EmbeddingBagClassifier(vocab_size, num_classes, embedding_dim, dropout)
    if name == "text_cnn":
        return TextCNNClassifier(
            vocab_size,
            num_classes,
            embedding_dim,
            kernel_sizes=[int(value) for value in params.get("kernel_sizes", [3, 4, 5])],
            channels=int(params.get("hidden_dim", 128)),
            dropout=dropout,
        )
    if name == "bilstm":
        return BiLSTMClassifier(
            vocab_size,
            num_classes,
            embedding_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            num_layers=int(params.get("num_layers", 2)),
            bidirectional=bool(params.get("bidirectional", True)),
            dropout=dropout,
        )
    raise ValueError(f"unknown model: {name}; choose from embedding_bag, text_cnn, bilstm")


def list_models() -> list[str]:
    return ["embedding_bag", "text_cnn", "bilstm"]
