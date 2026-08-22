from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "name": "ag_news",
        "data_dir": "data/raw",
        "manifest_dir": "data/manifests",
        "tokenizer": "simple_word",
        "vocab_size": 30000,
        "min_frequency": 2,
        "max_length": 128,
        "valid_ratio": 0.1,
        "num_workers": 0,
        "max_train_samples": None,
        "max_valid_samples": None,
        "max_test_samples": None,
    },
    "model": {"name": "embedding_bag", "embedding_dim": 128, "hidden_dim": 128, "dropout": 0.2},
    "train": {
        "epochs": 2,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "seed": 42,
        "amp": False,
        "grad_clip": 0.0,
        "best_metric": "macro_f1",
    },
    "device": "auto",
    "output_dir": "artifacts",
    "run_name": None,
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_value(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return parsed


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must use key=value: {item}")
        path, raw_value = item.split("=", 1)
        target = result
        parts = path.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _parse_value(raw_value)
    return result


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("config must be a YAML mapping")
            loaded = raw
    result = _merge(DEFAULT_CONFIG, loaded)
    return apply_overrides(result, overrides or [])


def show_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)
