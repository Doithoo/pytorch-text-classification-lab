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
