from pathlib import Path

import pytest
import torch

from text_classifier.config import load_config
from text_classifier.data.tokenizer import SimpleWordTokenizer
from text_classifier.inference import predict_text
from text_classifier.models import build_model
from text_classifier.training.checkpoint import load_checkpoint


def _write_checkpoint(path: Path) -> None:
    config = load_config(None, ["model.name=text_cnn", "device=cpu"])
    tokenizer = SimpleWordTokenizer.build(["hello world business sports"], 32, 1, 16)
    model = build_model("text_cnn", len(tokenizer.vocab), 4, config["model"])
    torch.save(
        {
            "schema_version": 1,
            "model_name": "text_cnn",
            "model_config": config["model"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scaler_state_dict": None,
            "epoch": 0,
            "metrics": {"macro_f1": 0.0},
            "config": config,
            "tokenizer_metadata": tokenizer.metadata(),
            "manifest_identity": "synthetic",
            "label_names": ["world", "sports", "business", "sci_tech"],
        },
        path,
    )


def test_predict_text_accepts_one_word_textcnn_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint)
    result = predict_text(checkpoint, "hello", top_k=2)
    assert result["label"] in {"world", "sports", "business", "sci_tech"}
    assert len(result["top_predictions"]) == 2
    assert set(result["probabilities"]) == {"world", "sports", "business", "sci_tech"}


def test_checkpoint_schema_is_validated(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    _write_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["schema_version"] = 99
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="unsupported checkpoint schema_version"):
        load_checkpoint(checkpoint)


def test_checkpoint_required_fields_are_validated(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing-fields.pt"
    torch.save({"schema_version": 1}, checkpoint)
    with pytest.raises(ValueError, match="missing required"):
        load_checkpoint(checkpoint)


def test_missing_checkpoint_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
