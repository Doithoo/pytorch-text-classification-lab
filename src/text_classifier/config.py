from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "name": "ag_news",
        "data_dir": "data/raw",
        "manifest_dir": "data/manifests",
        "tokenizer": "simple_word",
        "text_column": "text",
        "label_column": "label",
        "vocab_size": 30000,
        "min_frequency": 2,
        "max_length": 128,
        "valid_ratio": 0.1,
        "num_workers": 0,
        "max_train_samples": None,
        "max_valid_samples": None,
        "max_test_samples": None,
    },
    "model": {
        "name": "embedding_bag",
        "embedding_dim": 128,
        "hidden_dim": 128,
        "dropout": 0.2,
        "kernel_sizes": [3, 4, 5],
        "num_layers": 2,
        "bidirectional": True,
    },
    "train": {
        "epochs": 2,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "momentum": 0.9,
        "seed": 42,
        "amp": False,
        "deterministic": False,
        "grad_clip": 0.0,
        "best_metric": "macro_f1",
    },
    "device": "auto",
    "output_dir": "artifacts",
    "run_name": None,
}

_TOP_LEVEL_KEYS = {"data", "model", "train", "device", "output_dir", "run_name"}
_DATA_KEYS = set(DEFAULT_CONFIG["data"])
_MODEL_KEYS = {
    "name",
    "embedding_dim",
    "hidden_dim",
    "dropout",
    "kernel_sizes",
    "num_layers",
    "bidirectional",
}
_TRAIN_KEYS = set(DEFAULT_CONFIG["train"])
_BEST_METRICS = {"accuracy", "macro_f1", "macro_precision", "macro_recall"}


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


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], prefix: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        names = ", ".join(f"{prefix}.{name}" if prefix else name for name in unknown)
        raise ValueError(f"unknown configuration field(s): {names}")


def _require_int(mapping: dict[str, Any], key: str, minimum: int = 1) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _require_number(mapping: dict[str, Any], key: str, minimum: float = 0.0) -> float:
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(f"{key} must be a finite number >= {minimum}")
    return float(value)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a resolved configuration and return it unchanged."""
    _reject_unknown(config, _TOP_LEVEL_KEYS, "")
    data = _require_mapping(config, "data")
    model = _require_mapping(config, "model")
    train = _require_mapping(config, "train")
    _reject_unknown(data, _DATA_KEYS, "data")
    _reject_unknown(model, _MODEL_KEYS, "model")
    _reject_unknown(train, _TRAIN_KEYS, "train")

    if data.get("name") not in {"ag_news", "generic_csv"}:
        raise ValueError("data.name must be ag_news or generic_csv")
    if data.get("tokenizer") != "simple_word":
        raise ValueError("data.tokenizer must be simple_word")
    if not isinstance(data.get("data_dir"), str) or not data["data_dir"]:
        raise ValueError("data.data_dir must be a non-empty path string")
    if not isinstance(data.get("manifest_dir"), str) or not data["manifest_dir"]:
        raise ValueError("data.manifest_dir must be a non-empty path string")
    if not isinstance(data.get("text_column"), str) or not data["text_column"]:
        raise ValueError("data.text_column must be a non-empty string")
    if not isinstance(data.get("label_column"), str) or not data["label_column"]:
        raise ValueError("data.label_column must be a non-empty string")
    if data["name"] == "generic_csv" and data["text_column"] == data["label_column"]:
        raise ValueError("data.text_column and data.label_column must differ")
    _require_int(data, "vocab_size", 4)
    _require_int(data, "min_frequency")
    max_length = _require_int(data, "max_length", 2)
    valid_ratio = _require_number(data, "valid_ratio")
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("data.valid_ratio must be greater than 0 and less than 1")
    _require_int(data, "num_workers", 0)
    for name in ("max_train_samples", "max_valid_samples", "max_test_samples"):
        if data.get(name) is not None:
            _require_int(data, name)

    model_name = model.get("name")
    if model_name not in {"embedding_bag", "text_cnn", "bilstm"}:
        raise ValueError("model.name must be one of: embedding_bag, text_cnn, bilstm")
    _require_int(model, "embedding_dim")
    _require_int(model, "hidden_dim")
    dropout = _require_number(model, "dropout")
    if dropout >= 1.0:
        raise ValueError("model.dropout must be less than 1")
    if model_name == "text_cnn":
        kernel_sizes = model.get("kernel_sizes", [3, 4, 5])
        if not isinstance(kernel_sizes, list) or not kernel_sizes:
            raise ValueError("model.kernel_sizes must be a non-empty list")
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in kernel_sizes):
            raise ValueError("model.kernel_sizes values must be positive integers")
        if max(kernel_sizes) > max_length:
            raise ValueError("model.kernel_sizes cannot exceed data.max_length")
    if model_name == "bilstm":
        _require_int(model, "num_layers")
        if not isinstance(model.get("bidirectional"), bool):
            raise ValueError("model.bidirectional must be a boolean")

    _require_int(train, "epochs")
    _require_int(train, "batch_size")
    _require_number(train, "lr", minimum=1e-15)
    _require_number(train, "weight_decay")
    _require_number(train, "momentum")
    if train.get("optimizer") not in {"adamw", "adam", "sgd"}:
        raise ValueError("train.optimizer must be one of: adamw, adam, sgd")
    _require_int(train, "seed", 0)
    for name in ("amp", "deterministic"):
        if not isinstance(train.get(name), bool):
            raise ValueError(f"train.{name} must be a boolean")
    _require_number(train, "grad_clip")
    if train.get("best_metric") not in _BEST_METRICS:
        raise ValueError(f"train.best_metric must be one of: {', '.join(sorted(_BEST_METRICS))}")

    device = config.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", device):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<index>")
    if not isinstance(config.get("output_dir"), str) or not config["output_dir"]:
        raise ValueError("output_dir must be a non-empty path string")
    run_name = config.get("run_name")
    if run_name is not None and (
        not isinstance(run_name, str)
        or not run_name
        or Path(run_name).name != run_name
        or "/" in run_name
        or "\\" in run_name
    ):
        raise ValueError("run_name must be a single non-empty path component or null")
    return config


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("config must be a YAML mapping")
            loaded = raw
    result = _merge(DEFAULT_CONFIG, loaded)
    return validate_config(apply_overrides(result, overrides or []))


def show_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)
