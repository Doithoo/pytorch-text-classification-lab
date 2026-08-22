import json
from pathlib import Path

import pytest
import torch

from text_classifier.config import load_config
from text_classifier.data.tokenizer import SimpleWordTokenizer
from text_classifier.evaluation.compare import compare_runs
from text_classifier.inference.batch import predict_file
from text_classifier.models import build_model


def _write_run(root: Path, name: str, score: float, identity: str = "same", tokenizer: str = "tokenizer") -> Path:
    run = root / name
    run.mkdir()
    (run / "run.json").write_text(
        '{"manifest_identity": "'
        + identity
        + '", "tokenizer_sha256": "'
        + tokenizer
        + '", "label_names": ["A", "B"], "git_revision": "abc"}',
        encoding="utf-8",
    )
    (run / "metrics.csv").write_text(
        "epoch,valid_macro_f1\n1," + str(score - 0.1) + "\n2," + str(score) + "\n", encoding="utf-8"
    )
    return run


def _write_checkpoint(path: Path) -> Path:
    config = load_config(None, ["model.name=text_cnn", "device=cpu"])
    tokenizer = SimpleWordTokenizer.build(["hello world"], 32, 1, 16)
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
    return path


def test_compare_runs_validates_identity_and_sorts(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "first", 0.7)
    second = _write_run(tmp_path, "second", 0.8)
    rows = compare_runs([first, second])
    assert [row["run_dir"] for row in rows] == [str(second), str(first)]
    assert rows[0]["best_epoch"] == 2


def test_compare_runs_rejects_incompatible_identity(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "first", 0.7, "one")
    second = _write_run(tmp_path, "second", 0.8, "two")
    with pytest.raises(ValueError, match="manifest_identity"):
        compare_runs([first, second])


def test_compare_runs_rejects_incompatible_tokenizer(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "first", 0.7, tokenizer="one")
    second = _write_run(tmp_path, "second", 0.8, tokenizer="two")
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        compare_runs([first, second])


def test_predict_file_reads_csv_and_writes_jsonl(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt")
    source = tmp_path / "texts.csv"
    source.write_text("text\nhello\nworld\n", encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    assert predict_file(checkpoint, source, output, top_k=2) == 2
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["text"] == "hello"
    assert first["prediction"]["label"]
    with pytest.raises(FileExistsError, match="already exists"):
        predict_file(checkpoint, source, output, top_k=2)
