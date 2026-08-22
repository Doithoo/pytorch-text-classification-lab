import json
from pathlib import Path

import pytest

from text_classifier.data.manifest import audit_manifests, prepare_data


def _write_ag(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{(index % 4) + 1},title {index},description {index}" for index in range(rows)), encoding="utf-8"
    )


def test_prepare_data_creates_fixed_splits(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ag_news_csv"
    _write_ag(root / "train.csv", 20)
    _write_ag(root / "test.csv", 8)
    metadata = prepare_data(tmp_path / "raw", tmp_path / "manifests", valid_ratio=0.2, seed=42)
    assert metadata["counts"] == {"train": 16, "valid": 4, "test": 8}
    metadata_json = json.loads((tmp_path / "manifests" / "dataset.json").read_text())
    assert metadata_json["dataset"] == "ag_news"
    assert "adapter" not in metadata_json


@pytest.mark.parametrize("ratio", [-0.1, 0.0, 1.0, 1.1])
def test_prepare_data_rejects_invalid_validation_ratio(tmp_path: Path, ratio: float) -> None:
    with pytest.raises(ValueError, match="valid_ratio"):
        prepare_data(tmp_path / "raw", tmp_path / "manifests", valid_ratio=ratio, seed=42)


def test_generic_csv_adapter_creates_stable_multiclass_manifests(tmp_path: Path) -> None:
    raw = tmp_path / "generic"
    raw.mkdir()
    (raw / "train.csv").write_text(
        "text,label\n"
        + "\n".join(["alpha one,A", "alpha two,A", "beta one,B", "beta two,B", "gamma one,C", "gamma two,C"])
        + "\n",
        encoding="utf-8",
    )
    (raw / "test.csv").write_text("text,label\nunknown A,A\nunknown B,B\nunknown C,C\n", encoding="utf-8")
    metadata = prepare_data(raw, tmp_path / "manifests", valid_ratio=0.5, seed=7, dataset_name="generic_csv")
    assert metadata["labels"] == ["A", "B", "C"]
    assert metadata["adapter"] == "generic_csv"
    assert metadata["counts"] == {"train": 3, "valid": 3, "test": 3}
    audit = audit_manifests(tmp_path / "manifests", max_length=4)
    assert audit["duplicate_ids"] == 0
    assert audit["duplicate_text_across_splits"] == {
        "train_and_valid": 0,
        "train_and_test": 0,
        "valid_and_test": 0,
    }
    assert audit["tokenizer"]["coverage"]["test"]["oov_ratio"] > 0


def test_manifest_audit_detects_cross_split_duplicate_text(tmp_path: Path) -> None:
    root = tmp_path / "manifests"
    root.mkdir()
    rows = "id,text,label,label_id\n"
    (root / "train.csv").write_text(rows + "train-0,same,A,0\n", encoding="utf-8")
    (root / "valid.csv").write_text(rows + "valid-0,same,B,1\n", encoding="utf-8")
    (root / "test.csv").write_text(rows + "test-0,other,B,1\n", encoding="utf-8")
    (root / "dataset.json").write_text('{"schema_version": 1, "labels": ["A", "B"]}', encoding="utf-8")
    audit = audit_manifests(root)
    assert audit["duplicate_text_across_splits"]["train_and_valid"] == 1
    assert audit["conflicting_duplicate_labels"] == 1


def test_generic_csv_rejects_test_only_label(tmp_path: Path) -> None:
    raw = tmp_path / "generic"
    raw.mkdir()
    (raw / "train.csv").write_text("text,label\na1,A\na2,A\nb1,B\nb2,B\n", encoding="utf-8")
    (raw / "test.csv").write_text("text,label\nc1,C\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the training labels"):
        prepare_data(raw, tmp_path / "manifests", dataset_name="generic_csv")
