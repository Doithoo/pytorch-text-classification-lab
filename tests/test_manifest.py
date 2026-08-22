import json
from pathlib import Path

import pytest

from text_classifier.data.manifest import prepare_data


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
    assert json.loads((tmp_path / "manifests" / "dataset.json").read_text())["dataset"] == "ag_news"


@pytest.mark.parametrize("ratio", [-0.1, 0.0, 1.0, 1.1])
def test_prepare_data_rejects_invalid_validation_ratio(tmp_path: Path, ratio: float) -> None:
    with pytest.raises(ValueError, match="valid_ratio"):
        prepare_data(tmp_path / "raw", tmp_path / "manifests", valid_ratio=ratio, seed=42)
