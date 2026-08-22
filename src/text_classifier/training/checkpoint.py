from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import DEFAULT_CONFIG, _merge, validate_config
from ..data.manifest import LABEL_NAMES

CHECKPOINT_SCHEMA_VERSION = 1
_REQUIRED_FIELDS = {
    "schema_version",
    "model_name",
    "model_config",
    "model_state_dict",
    "epoch",
    "config",
    "tokenizer_metadata",
    "manifest_identity",
}


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate a trusted project checkpoint.

    PyTorch checkpoints use pickle internally. Callers must not load files from
    untrusted sources.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint_path}")
    missing = sorted(_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing required field(s): {', '.join(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version {payload['schema_version']!r}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    config = payload["config"]
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a mapping")
    normalized_config = validate_config(_merge(DEFAULT_CONFIG, config))
    if payload["model_name"] != normalized_config["model"]["name"]:
        raise ValueError("checkpoint model_name does not match checkpoint config")
    if not isinstance(payload["model_state_dict"], dict):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    if not isinstance(payload["epoch"], int) or payload["epoch"] < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    if not isinstance(payload["manifest_identity"], str) or not payload["manifest_identity"]:
        raise ValueError("checkpoint manifest_identity must be a non-empty string")
    payload["config"] = normalized_config
    payload["model_config"] = normalized_config["model"]
    tokenizer = payload["tokenizer_metadata"]
    vocab = tokenizer.get("vocab") if isinstance(tokenizer, dict) else None
    if not isinstance(vocab, dict):
        raise ValueError("checkpoint tokenizer_metadata.vocab must be a mapping")
    for token in ("<pad>", "<unk>", "<bos>", "<eos>"):
        if not isinstance(vocab.get(token), int):
            raise ValueError(f"checkpoint vocabulary is missing integer token {token}")
    if not isinstance(tokenizer.get("max_length"), int) or tokenizer["max_length"] < 2:
        raise ValueError("checkpoint tokenizer_metadata.max_length must be an integer >= 2")
    labels = payload.get("label_names", LABEL_NAMES)
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("checkpoint label_names must be a non-empty string list")
    payload["label_names"] = labels
    return payload
