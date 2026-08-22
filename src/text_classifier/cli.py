from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config, show_config
from .data.dataset import TextClassificationDataset, make_collate_fn
from .data.manifest import prepare_data
from .data.tokenizer import SimpleWordTokenizer
from .models import build_model, list_models
from .training.train import evaluate_loader, resolve_device, train


def _config_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])


def cmd_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides)
    metadata = prepare_data(
        config["data"]["data_dir"],
        config["data"]["manifest_dir"],
        float(config["data"]["valid_ratio"]),
        int(config["train"]["seed"]),
    )
    print(json.dumps(metadata, indent=2))


def cmd_inspect(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides)
    manifest_dir = Path(config["data"]["manifest_dir"])
    for split in ("train", "valid", "test"):
        path = manifest_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        dataset = TextClassificationDataset(
            str(path), SimpleWordTokenizer({"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}, 128)
        )
        lengths = [len(dataset[index]["input_ids"]) for index in range(len(dataset))]
        print(f"{split}: samples={len(dataset)} min_tokens={min(lengths)} max_tokens={max(lengths)}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer = SimpleWordTokenizer(
        checkpoint["tokenizer_metadata"]["vocab"], int(checkpoint["tokenizer_metadata"]["max_length"])
    )
    model = build_model(config["model"]["name"], len(tokenizer.vocab), 4, config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device or config["device"])
    model.to(device)
    dataset = TextClassificationDataset(str(Path(config["data"]["manifest_dir"]) / f"{args.split}.csv"), tokenizer)
    loader = DataLoader(
        dataset, batch_size=int(config["train"]["batch_size"]), collate_fn=make_collate_fn(tokenizer.pad_id)
    )
    metrics, errors = evaluate_loader(model, loader, device, 4, return_errors=True)
    output = Path(args.output or Path(args.checkpoint).parent / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer = SimpleWordTokenizer(
        checkpoint["tokenizer_metadata"]["vocab"], int(checkpoint["tokenizer_metadata"]["max_length"])
    )
    model = build_model(config["model"]["name"], len(tokenizer.vocab), 4, config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ids = torch.tensor([tokenizer.encode(args.text)], dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        probabilities = torch.softmax(model(ids, mask), dim=-1)[0]
    index = int(probabilities.argmax())
    labels = ["world", "sports", "business", "sci_tech"]
    print(
        json.dumps(
            {
                "label": labels[index],
                "confidence": float(probabilities[index]),
                "probabilities": probabilities.tolist(),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="text-classify")
    parser.add_argument("--version", action="version", version="0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in [
        ("prepare-data", cmd_prepare, "create audited train/valid/test manifests"),
        ("inspect-data", cmd_inspect, "inspect prepared text lengths"),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        _config_parser(sub)
        sub.set_defaults(handler=handler)
    train_parser = subparsers.add_parser("train", help="train a classifier")
    _config_parser(train_parser)
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.set_defaults(
        handler=lambda args: train(
            load_config(args.config, args.overrides), args.dry_run, str(args.resume) if args.resume else None
        )
    )
    eval_parser = subparsers.add_parser("evaluate", help="evaluate a checkpoint")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", choices=["valid", "test"], default="test")
    eval_parser.add_argument("--device", default=None)
    eval_parser.add_argument("--output", default=None)
    eval_parser.set_defaults(handler=cmd_evaluate)
    predict_parser = subparsers.add_parser("predict", help="classify one text")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--text", required=True)
    predict_parser.set_defaults(handler=cmd_predict)
    config_parser = subparsers.add_parser("show-config", help="print resolved configuration")
    _config_parser(config_parser)
    config_parser.set_defaults(handler=lambda args: print(show_config(load_config(args.config, args.overrides))))
    models_parser = subparsers.add_parser("list-models", help="list built-in models")
    models_parser.set_defaults(handler=lambda args: print("\n".join(list_models())))
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main(sys.argv[1:])
