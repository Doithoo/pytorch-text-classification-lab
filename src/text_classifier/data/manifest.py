from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .tokenizer import SimpleWordTokenizer

LABEL_NAMES = ["world", "sports", "business", "sci_tech"]
MANIFEST_COLUMNS = ["id", "text", "label", "label_id"]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _read_ag_csv(path: Path, prefix: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if len(row) < 3:
                raise ValueError(f"expected label,title,description in {path}, row {index + 1}")
            try:
                label_id = int(row[0]) - 1
            except ValueError as exc:
                raise ValueError(f"AG News label must be an integer in {path}, row {index + 1}") from exc
            if label_id not in range(len(LABEL_NAMES)):
                raise ValueError(f"AG News label must be in 1..4, got {row[0]}")
            text = f"{row[1]} {row[2]}".strip()
            if not text:
                raise ValueError(f"empty text in {path}, row {index + 1}")
            rows.append(
                {"id": f"{prefix}-{index:06d}", "text": text, "label": LABEL_NAMES[label_id], "label_id": label_id}
            )
    return rows, list(LABEL_NAMES)


def _read_generic_csv(path: Path, prefix: str, text_column: str, label_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in (text_column, label_column) if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        for index, raw in enumerate(reader):
            text = str(raw.get(text_column, "")).strip()
            label = str(raw.get(label_column, "")).strip()
            if not text or not label:
                raise ValueError(f"text and label must be non-empty in {path}, row {index + 2}")
            rows.append({"id": f"{prefix}-{index:06d}", "text": text, "label": label})
    return rows


def _assign_label_ids(rows: list[dict[str, Any]], labels: list[str]) -> None:
    label_to_id = {label: index for index, label in enumerate(labels)}
    for row in rows:
        if row["label"] not in label_to_id:
            raise ValueError(f"label {row['label']!r} is absent from the training labels")
        row["label_id"] = label_to_id[row["label"]]


def _split_rows(
    rows: list[dict[str, Any]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label_id"])].append(row)
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for label in sorted(by_label):
        label_rows = by_label[label]
        if len(label_rows) < 2:
            raise ValueError(f"training label {label} needs at least two rows for a train/valid split")
        rng.shuffle(label_rows)
        count = min(len(label_rows) - 1, max(1, round(len(label_rows) * valid_ratio)))
        valid.extend(label_rows[:count])
        train.extend(label_rows[count:])
    train.sort(key=lambda row: str(row["id"]))
    valid.sort(key=lambda row: str(row["id"]))
    return train, valid


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(
    data_dir: str | Path,
    manifest_dir: str | Path,
    valid_ratio: float = 0.1,
    seed: int = 42,
    dataset_name: str = "ag_news",
    text_column: str = "text",
    label_column: str = "label",
) -> dict[str, Any]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be greater than 0 and less than 1")
    if dataset_name not in {"ag_news", "generic_csv"}:
        raise ValueError("dataset_name must be ag_news or generic_csv")
    root = Path(data_dir) / "ag_news_csv" if dataset_name == "ag_news" else Path(data_dir)
    train_path, test_path = root / "train.csv", root / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"expected {train_path} and {test_path}")
    if dataset_name == "ag_news":
        source_train, labels = _read_ag_csv(train_path, "train")
        source_test, _ = _read_ag_csv(test_path, "test")
    else:
        source_train = _read_generic_csv(train_path, "train", text_column, label_column)
        source_test = _read_generic_csv(test_path, "test", text_column, label_column)
        labels = sorted({str(row["label"]) for row in source_train})
        if len(labels) < 2:
            raise ValueError("generic_csv requires at least two training labels")
        _assign_label_ids(source_train, labels)
        _assign_label_ids(source_test, labels)
    train, valid = _split_rows(source_train, valid_ratio, seed)
    source_test.sort(key=lambda row: str(row["id"]))
    destination = Path(manifest_dir)
    _write_manifest(destination / "train.csv", train)
    _write_manifest(destination / "valid.csv", valid)
    _write_manifest(destination / "test.csv", source_test)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset_name,
        "labels": labels,
        "seed": seed,
        "valid_ratio": valid_ratio,
        "counts": {"train": len(train), "valid": len(valid), "test": len(source_test)},
        "source_sha256": {"train.csv": _hash_file(train_path), "test.csv": _hash_file(test_path)},
        "manifest_sha256": {name: _hash_file(destination / f"{name}.csv") for name in ("train", "valid", "test")},
    }
    if dataset_name == "generic_csv":
        metadata.update({"adapter": "generic_csv", "text_column": text_column, "label_column": label_column})
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
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise ValueError(f"manifest {path} must have columns: {', '.join(MANIFEST_COLUMNS)}")
        rows = list(reader)
    if limit is not None:
        rows = rows[: int(limit)]
    for row in rows:
        if not row["id"] or not row["text"] or not row["label"]:
            raise ValueError(f"manifest {path} contains an empty id, text, or label")
        try:
            row["label_id"] = int(row["label_id"])
        except ValueError as exc:
            raise ValueError(f"manifest {path} contains a non-integer label_id") from exc
    return rows


def load_manifest_metadata(manifest_dir: str | Path) -> dict[str, Any]:
    path = Path(manifest_dir) / "dataset.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported dataset metadata in {path}")
    labels = metadata.get("labels")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"dataset labels must be a non-empty string list in {path}")
    return metadata


def manifest_identity(manifest_dir: str | Path) -> str:
    metadata = load_manifest_metadata(manifest_dir)
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def audit_manifests(
    manifest_dir: str | Path,
    max_length: int = 128,
    vocab_size: int = 30000,
    min_frequency: int = 2,
) -> dict[str, Any]:
    root = Path(manifest_dir)
    split_rows = {split: load_manifest(root / f"{split}.csv") for split in ("train", "valid", "test")}
    all_rows = [row for rows in split_rows.values() for row in rows]
    normalized = {split: Counter(_normalize_text(row["text"]) for row in rows) for split, rows in split_rows.items()}
    duplicate_texts = {
        split: sum(count - 1 for count in counts.values() if count > 1) for split, counts in normalized.items()
    }
    cross_split: dict[str, int] = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        cross_split[f"{left}_and_{right}"] = len(set(normalized[left]) & set(normalized[right]))
    ids = [str(row["id"]) for row in all_rows]
    lengths = [len(SimpleWordTokenizer.tokenize(str(row["text"]))) + 2 for row in all_rows]
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        labels_by_text[_normalize_text(str(row["text"]))].add(str(row["label"]))
    tokenizer = SimpleWordTokenizer.build(
        [str(row["text"]) for row in split_rows["train"]], vocab_size, min_frequency, max_length
    )
    token_coverage: dict[str, dict[str, float | int]] = {}
    for split, rows in split_rows.items():
        tokens = [token for row in rows for token in SimpleWordTokenizer.tokenize(str(row["text"]))]
        unknown = sum(token not in tokenizer.vocab for token in tokens)
        token_coverage[split] = {
            "tokens": len(tokens),
            "unknown_tokens": unknown,
            "oov_ratio": unknown / max(len(tokens), 1),
        }
    labels = load_manifest_metadata(root)["labels"]
    for row in all_rows:
        label_id = int(row["label_id"])
        if label_id < 0 or label_id >= len(labels) or labels[label_id] != row["label"]:
            raise ValueError(f"manifest label mapping is inconsistent for id {row['id']}")
    label_counts = {
        split: dict(sorted(Counter(row["label"] for row in rows).items())) for split, rows in split_rows.items()
    }
    return {
        "schema_version": 1,
        "manifest_identity": manifest_identity(root),
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "labels": labels,
        "label_counts": label_counts,
        "empty_text": sum(not str(row["text"]).strip() for row in all_rows),
        "duplicate_ids": len(ids) - len(set(ids)),
        "conflicting_duplicate_labels": sum(len(values) > 1 for values in labels_by_text.values()),
        "duplicate_text_within_split": duplicate_texts,
        "duplicate_text_across_splits": cross_split,
        "tokenizer": {
            "vocab_size": len(tokenizer.vocab),
            "min_frequency": min_frequency,
            "coverage": token_coverage,
        },
        "lengths": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "truncated_at_max_length": sum(length > max_length for length in lengths),
        },
    }
