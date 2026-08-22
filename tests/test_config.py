from pathlib import Path

import pytest

from text_classifier.config import apply_overrides, load_config


def test_nested_overrides() -> None:
    config = load_config(None, ["train.batch_size=7", "model.name=text_cnn", "train.amp=true"])
    assert config["train"]["batch_size"] == 7
    assert config["model"]["name"] == "text_cnn"
    assert config["train"]["amp"] is True


def test_merge_does_not_mutate_input() -> None:
    original = {"a": {"b": 1}}
    changed = apply_overrides(original, ["a.b=2"])
    assert original["a"]["b"] == 1
    assert changed["a"]["b"] == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("data.valid_ratio=0", "valid_ratio"),
        ("data.valid_ratio=1", "valid_ratio"),
        ("train.optimizer=rmsprop", "optimizer"),
        ("train.batch_size=0", "batch_size"),
        ("train.lr=.inf", "lr"),
        ("model.dropout=1", "dropout"),
        ("data.name=unknown", "data.name"),
        ("unknown.value=1", "unknown configuration"),
    ],
)
def test_invalid_config_is_rejected(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(None, [override])


def test_generic_csv_config_is_supported() -> None:
    config = load_config(None, ["data.name=generic_csv", "data.text_column=body", "data.label_column=category"])
    assert config["data"]["name"] == "generic_csv"
    assert config["data"]["text_column"] == "body"


def test_run_name_cannot_escape_output_directory() -> None:
    with pytest.raises(ValueError, match="run_name"):
        load_config(None, ["run_name=../outside"])


def test_config_file_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)
