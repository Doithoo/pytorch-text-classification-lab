from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import TextClassificationDataset, make_collate_fn
from ..data.manifest import load_manifest, load_manifest_metadata, manifest_identity
from ..data.tokenizer import SimpleWordTokenizer
from ..evaluation.metrics import classification_metrics, save_json
from ..models import build_model
from .checkpoint import load_checkpoint


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def _build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_config = config["train"]
    learning_rate = float(train_config["lr"])
    weight_decay = float(train_config["weight_decay"])
    name = str(train_config["optimizer"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(train_config["momentum"]),
            weight_decay=weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def _loader(config: dict[str, Any], split: str, tokenizer: SimpleWordTokenizer, shuffle: bool) -> DataLoader:
    data = config["data"]
    limit = data.get(f"max_{split}_samples")
    dataset = TextClassificationDataset(str(Path(data["manifest_dir"]) / f"{split}.csv"), tokenizer, limit)
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data.get("num_workers", 0)),
        collate_fn=make_collate_fn(tokenizer.pad_id),
        pin_memory=torch.cuda.is_available(),
    )


def _build_tokenizer(config: dict[str, Any]) -> SimpleWordTokenizer:
    manifest_path = Path(config["data"]["manifest_dir"]) / "train.csv"
    rows = load_manifest(manifest_path, config["data"].get("max_train_samples"))
    return SimpleWordTokenizer.build(
        [row["text"] for row in rows],
        vocab_size=int(config["data"]["vocab_size"]),
        min_frequency=int(config["data"]["min_frequency"]),
        max_length=int(config["data"]["max_length"]),
    )


def _forward(model: nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids, attention_mask)
    return logits, labels


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    return_errors: bool = False,
    label_names: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    errors: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            logits, target = _forward(model, batch, device)
            probabilities = torch.softmax(logits, dim=-1)
            predicted = probabilities.argmax(dim=-1)
            labels.extend(target.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            if return_errors:
                for sample_id, text, true, pred, confidence in zip(
                    batch["ids"],
                    batch["texts"],
                    target.cpu().tolist(),
                    predicted.cpu().tolist(),
                    probabilities.max(dim=-1).values.cpu().tolist(),
                    strict=True,
                ):
                    if true != pred:
                        row: dict[str, Any] = {
                            "id": sample_id,
                            "text": text,
                            "true": true,
                            "pred": pred,
                            "confidence": confidence,
                        }
                        if label_names is not None:
                            row["true_label"] = label_names[true]
                            row["pred_label"] = label_names[pred]
                        errors.append(row)
    metrics = classification_metrics(labels, predictions, num_classes)
    if label_names is not None:
        metrics["label_names"] = label_names
    errors.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return metrics, errors


def _save_metrics_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    tokenizer: SimpleWordTokenizer,
    epoch: int,
    metrics: dict[str, Any],
    label_names: list[str],
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": config["model"]["name"],
        "model_config": config["model"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "label_names": label_names,
        "config": config,
        "tokenizer_metadata": tokenizer.metadata(),
        "manifest_identity": manifest_identity(config["data"]["manifest_dir"]),
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_resume_config(config: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    previous = checkpoint["config"]
    if config["model"] != previous["model"]:
        raise ValueError("resume requires the same model configuration as the checkpoint")
    for key in ("name", "tokenizer", "vocab_size", "min_frequency", "max_length"):
        if config["data"][key] != previous["data"][key]:
            raise ValueError(f"resume requires unchanged data.{key}")
    for key in ("optimizer", "lr", "weight_decay", "momentum", "best_metric"):
        if config["train"][key] != previous["train"][key]:
            raise ValueError(f"resume requires unchanged train.{key}")


def train(config: dict[str, Any], dry_run: bool = False, resume: str | None = None) -> Path | None:
    started = time.time()
    set_seed(int(config["train"]["seed"]), bool(config["train"]["deterministic"]))
    device = resolve_device(str(config["device"]))
    checkpoint: dict[str, Any] | None = None
    if resume:
        checkpoint = load_checkpoint(resume, map_location=device)
        _validate_resume_config(config, checkpoint)
        expected_identity = manifest_identity(config["data"]["manifest_dir"])
        if checkpoint["manifest_identity"] != expected_identity:
            raise ValueError("resume checkpoint manifest identity does not match current prepared data")
        tokenizer_metadata = checkpoint["tokenizer_metadata"]
        tokenizer = SimpleWordTokenizer(tokenizer_metadata["vocab"], int(tokenizer_metadata["max_length"]))
    else:
        tokenizer = _build_tokenizer(config)
    metadata = load_manifest_metadata(config["data"]["manifest_dir"])
    label_names = [str(label) for label in metadata["labels"]]
    num_classes = len(label_names)
    model = build_model(str(config["model"]["name"]), len(tokenizer.vocab), num_classes, config["model"])
    model.to(device)
    optimizer = _build_optimizer(model, config)
    amp_enabled = bool(config["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    if checkpoint is not None:
        if checkpoint.get("optimizer_state_dict") is None:
            raise ValueError("resume checkpoint does not contain optimizer_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
    train_loader = _loader(config, "train", tokenizer, shuffle=True)
    valid_loader = _loader(config, "valid", tokenizer, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    if dry_run:
        batch = next(iter(train_loader))
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits, labels = _forward(model, batch, device)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        print(
            f"device={device} amp={amp_enabled} input_ids={tuple(batch['input_ids'].shape)} logits={tuple(logits.shape)} loss={loss.item():.6f}"
        )
        print("dry-run OK")
        return None
    if resume:
        run_dir = Path(resume).parent
    else:
        run_name = config.get("run_name") or time.strftime("run-%Y%m%d-%H%M%S")
        run_dir = Path(config["output_dir"]) / str(run_name)
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}; choose another run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    tokenizer.save(run_dir / "tokenizer.json")
    history: list[dict[str, Any]] = []
    history_path = run_dir / "metrics.csv"
    if resume and history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
    best_metric = max(
        [float(row.get(f"valid_{config['train']['best_metric']}", -float("inf"))) for row in history],
        default=-float("inf"),
    )
    if checkpoint is not None:
        best_metric = max(
            best_metric, float(checkpoint.get("metrics", {}).get(config["train"]["best_metric"], -float("inf")))
        )
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, labels = _forward(model, batch, device)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if float(config["train"]["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            batches += 1
        valid_metrics, _ = evaluate_loader(model, valid_loader, device, num_classes)
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(batches, 1),
            **{f"valid_{k}": v for k, v in valid_metrics.items() if isinstance(v, (int, float))},
        }
        history.append(record)
        print(json.dumps(record))
        payload = _checkpoint(model, optimizer, config, tokenizer, epoch, valid_metrics, label_names, scaler)
        torch.save(payload, run_dir / "last.pt")
        if float(valid_metrics[config["train"]["best_metric"]]) > best_metric:
            best_metric = float(valid_metrics[config["train"]["best_metric"]])
            torch.save(payload, run_dir / "best.pt")
    _save_metrics_csv(run_dir / "metrics.csv", history)
    tokenizer_metadata = tokenizer.metadata()
    resolved_manifest_identity = manifest_identity(config["data"]["manifest_dir"])
    save_json(
        run_dir / "run.json",
        {
            "schema_version": 2,
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(),
            "command": sys.argv,
            "elapsed_seconds": round(time.time() - started, 3),
            "manifest_identity": resolved_manifest_identity,
            "config_sha256": _sha256_json(config),
            "tokenizer_sha256": _sha256_json(tokenizer_metadata),
            "uv_lock_sha256": _file_sha256(Path("uv.lock")),
            "label_names": label_names,
            "best_metric_name": config["train"]["best_metric"],
            "best_metric": best_metric,
            "deterministic": bool(config["train"]["deterministic"]),
        },
    )
    return run_dir
