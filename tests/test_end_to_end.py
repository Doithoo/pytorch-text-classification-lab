import copy
import csv
from pathlib import Path

import pytest

from text_classifier.cli import main
from text_classifier.config import load_config
from text_classifier.data.manifest import prepare_data
from text_classifier.training.checkpoint import load_checkpoint
from text_classifier.training.train import train


def _write_ag(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index in range(rows):
            writer.writerow([(index % 4) + 1, f"title {index}", f"description {index}"])


def test_generic_csv_three_class_training(tmp_path: Path) -> None:
    raw = tmp_path / "generic"
    raw.mkdir()
    with (raw / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for label in ("alpha", "beta", "gamma"):
            for index in range(6):
                writer.writerow([f"{label} example {index}", label])
    with (raw / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        for label in ("alpha", "beta", "gamma"):
            writer.writerow([f"{label} held out", label])
    manifests = tmp_path / "generic-manifests"
    prepare_data(raw, manifests, valid_ratio=0.25, seed=3, dataset_name="generic_csv")
    config = load_config(
        None,
        [
            "data.name=generic_csv",
            f"data.data_dir={raw}",
            f"data.manifest_dir={manifests}",
            "data.min_frequency=1",
            "train.epochs=1",
            "train.batch_size=3",
            "model.name=embedding_bag",
            "device=cpu",
            f"output_dir={tmp_path / 'artifacts'}",
            "run_name=generic-run",
        ],
    )
    run_dir = train(config)
    assert run_dir is not None
    payload = load_checkpoint(run_dir / "best.pt")
    assert payload["label_names"] == ["alpha", "beta", "gamma"]


def test_cpu_dry_run(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "ag_news_csv"
    _write_ag(raw / "train.csv", 40)
    _write_ag(raw / "test.csv", 16)
    manifests = tmp_path / "manifests"
    prepare_data(raw.parent, manifests, valid_ratio=0.2, seed=42)
    main(["inspect-data", "--set", f"data.manifest_dir={manifests}"])
    assert (manifests / "inspection.json").exists()
    config = load_config(
        None,
        [
            f"data.manifest_dir={manifests}",
            "data.max_train_samples=16",
            "data.max_valid_samples=8",
            "data.max_test_samples=8",
            "train.batch_size=4",
            "model.name=embedding_bag",
            "device=cpu",
        ],
    )
    assert train(config, dry_run=True) is None
    config["train"]["epochs"] = 1
    config["run_name"] = "test-run"
    config["output_dir"] = str(tmp_path / "artifacts")
    run_dir = train(config)
    assert run_dir is not None
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "metrics.csv").exists()

    incompatible = copy.deepcopy(config)
    incompatible["train"]["epochs"] = 2
    incompatible["train"]["lr"] = 0.5
    with pytest.raises(ValueError, match="train.lr"):
        train(incompatible, resume=str(run_dir / "last.pt"))

    config["train"]["epochs"] = 2
    resumed_dir = train(config, resume=str(run_dir / "last.pt"))
    assert resumed_dir == run_dir
    assert len((run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()) == 3

    checkpoint = run_dir / "best.pt"
    main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--manifest-dir",
            str(manifests),
            "--device",
            "cpu",
        ]
    )
    evaluation_dir = run_dir / "evaluation" / "test"
    assert (evaluation_dir / "metrics.json").exists()
    assert (evaluation_dir / "errors.jsonl").exists()
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--manifest-dir",
                str(manifests),
                "--device",
                "cpu",
            ]
        )
    main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--manifest-dir",
            str(manifests),
            "--device",
            "cpu",
            "--overwrite",
        ]
    )
    main(["predict", "--checkpoint", str(checkpoint), "--text", "hello", "--top-k", "2"])
