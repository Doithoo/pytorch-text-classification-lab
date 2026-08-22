from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

LABEL_NAMES = ["world", "sports", "business", "sci_tech"]


def _read_ag_csv(path: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if len(row) < 3:
                raise ValueError(f"expected label,title,description in {path}, row {index + 1}")
            label = int(row[0]) - 1
            if label not in range(len(LABEL_NAMES)):
                raise ValueError(f"AG News label must be in 1..4, got {row[0]}")
            text = f"{row[1]} {row[2]}".strip()
            rows.append({"id": f"{prefix}-{index:06d}", "text": text, "label": LABEL_NAMES[label], "label_id": label})
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "label", "label_id"])
        writer.writeheader()
        writer.writerows(rows)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(
    data_dir: str | Path, manifest_dir: str | Path, valid_ratio: float = 0.1, seed: int = 42
) -> dict[str, Any]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be greater than 0 and less than 1")
    root = Path(data_dir) / "ag_news_csv"
    train_path, test_path = root / "train.csv", root / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"expected {train_path} and {test_path}; run scripts/download_data.py first")
    source_train = _read_ag_csv(train_path, "train")
    source_test = _read_ag_csv(test_path, "test")
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source_train:
        by_label[int(row["label_id"])].append(row)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for label in sorted(by_label):
        rows = by_label[label]
        rng.shuffle(rows)
        count = max(1, round(len(rows) * valid_ratio))
        valid.extend(rows[:count])
        train.extend(rows[count:])
    train.sort(key=lambda row: str(row["id"]))
    valid.sort(key=lambda row: str(row["id"]))
    source_test.sort(key=lambda row: str(row["id"]))
    destination = Path(manifest_dir)
    _write_manifest(destination / "train.csv", train)
    _write_manifest(destination / "valid.csv", valid)
    _write_manifest(destination / "test.csv", source_test)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "ag_news",
        "labels": LABEL_NAMES,
        "seed": seed,
        "valid_ratio": valid_ratio,
        "counts": {"train": len(train), "valid": len(valid), "test": len(source_test)},
        "source_sha256": {"train.csv": _hash_file(train_path), "test.csv": _hash_file(test_path)},
        "manifest_sha256": {name: _hash_file(destination / f"{name}.csv") for name in ("train", "valid", "test")},
    }
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "summary.txt").write_text(
        "\n".join([f"{key}={value}" for key, value in metadata["counts"].items()])
        + "\n"
        + json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def load_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[: int(limit)]
    for row in rows:
        row["label_id"] = int(row["label_id"])
    return rows


def manifest_identity(manifest_dir: str | Path) -> str:
    metadata = load_manifest_metadata(manifest_dir)
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def load_manifest_metadata(manifest_dir: str | Path) -> dict[str, Any]:
    path = Path(manifest_dir) / "dataset.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported dataset metadata in {path}")
    labels = metadata.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"dataset labels must be a non-empty string list in {path}")
    return metadata
