from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from . import __version__
from .config import load_config, show_config
from .data.dataset import TextClassificationDataset, make_collate_fn
from .data.manifest import audit_manifests, load_manifest, load_manifest_metadata, manifest_identity, prepare_data
from .data.tokenizer import SimpleWordTokenizer
from .evaluation.compare import compare_runs
from .evaluation.metrics import save_json
from .inference import export_inference_checkpoint, predict_file, predict_text
from .models import build_model, list_models
from .training.checkpoint import load_checkpoint
from .training.train import evaluate_loader, resolve_device, train


def _config_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML configuration path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")


def cmd_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides)
    metadata = prepare_data(
        config["data"]["data_dir"],
        config["data"]["manifest_dir"],
        float(config["data"]["valid_ratio"]),
        int(config["train"]["seed"]),
        str(config["data"]["name"]),
        str(config["data"]["text_column"]),
        str(config["data"]["label_column"]),
    )
    print(json.dumps(metadata, indent=2))


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def cmd_inspect(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.overrides)
    manifest_dir = Path(config["data"]["manifest_dir"])
    max_length = int(config["data"]["max_length"])
    for split in ("train", "valid", "test"):
        path = manifest_dir / f"{split}.csv"
        rows = load_manifest(path)
        if not rows:
            raise ValueError(f"manifest split is empty: {path}")
        lengths = [len(SimpleWordTokenizer.tokenize(str(row["text"]))) + 2 for row in rows]
        labels = Counter(str(row["label"]) for row in rows)
        truncated = sum(length > max_length for length in lengths)
        print(
            f"{split}: samples={len(rows)} min={min(lengths)} p50={_percentile(lengths, 0.5)} "
            f"p95={_percentile(lengths, 0.95)} max={max(lengths)} "
            f"truncated={truncated} labels={dict(sorted(labels.items()))}"
        )
    audit = audit_manifests(
        manifest_dir,
        max_length,
        int(config["data"]["vocab_size"]),
        int(config["data"]["min_frequency"]),
    )
    output = Path(args.output) if args.output else manifest_dir / "inspection.json"
    save_json(output, audit)
    print(f"audit: {output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    labels = checkpoint["label_names"]
    tokenizer_metadata = checkpoint["tokenizer_metadata"]
    tokenizer = SimpleWordTokenizer(
        {str(key): int(value) for key, value in tokenizer_metadata["vocab"].items()},
        int(tokenizer_metadata["max_length"]),
    )
    model = build_model(config["model"]["name"], len(tokenizer.vocab), len(labels), config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    model.to(device)
    manifest_dir = Path(args.manifest_dir or config["data"]["manifest_dir"])
    if (manifest_dir / "dataset.json").is_file():
        current_identity = manifest_identity(manifest_dir)
        if current_identity != checkpoint["manifest_identity"]:
            raise ValueError("evaluation manifest identity does not match checkpoint")
        metadata = load_manifest_metadata(manifest_dir)
        if metadata["labels"] != labels:
            raise ValueError("evaluation label order does not match checkpoint")
    dataset = TextClassificationDataset(str(manifest_dir / f"{args.split}.csv"), tokenizer)
    if not dataset:
        raise ValueError(f"evaluation split is empty: {args.split}")
    loader = DataLoader(
        dataset, batch_size=int(config["train"]["batch_size"]), collate_fn=make_collate_fn(tokenizer.pad_id)
    )
    metrics, errors = evaluate_loader(model, loader, device, len(labels), return_errors=True, label_names=labels)
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluation" / args.split
    existing = [output / "metrics.json", output / "errors.jsonl"]
    if not args.overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"evaluation output already exists: {output}; pass --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "metrics.json", metrics)
    (output / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    result = predict_text(args.checkpoint, args.text, args.device, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_predict_file(args: argparse.Namespace) -> None:
    count = predict_file(args.checkpoint, args.input, args.output, args.device, args.top_k, args.overwrite)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "predictions": count}, indent=2))


def cmd_export_inference(args: argparse.Namespace) -> None:
    weights, metadata = export_inference_checkpoint(args.checkpoint, args.output, args.overwrite)
    print(json.dumps({"weights": str(weights), "metadata": str(metadata)}, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    rows = compare_runs(args.runs, args.metric)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text-classify")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-data", help="create audited train/valid/test manifests")
    _config_parser(prepare_parser)
    prepare_parser.set_defaults(handler=cmd_prepare)
    inspect_parser = subparsers.add_parser("inspect-data", help="inspect and audit prepared text")
    _config_parser(inspect_parser)
    inspect_parser.add_argument("--output", type=Path, default=None)
    inspect_parser.set_defaults(handler=cmd_inspect)

    train_parser = subparsers.add_parser("train", help="train a classifier")
    _config_parser(train_parser)
    train_parser.add_argument("--dry-run", action="store_true", help="run one forward/backward batch without outputs")
    train_parser.add_argument("--resume", type=Path, default=None, help="trusted last.pt checkpoint")
    train_parser.set_defaults(
        handler=lambda args: train(
            load_config(args.config, args.overrides), args.dry_run, str(args.resume) if args.resume else None
        )
    )

    eval_parser = subparsers.add_parser("evaluate", help="evaluate a trusted checkpoint")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", choices=["valid", "test"], default="test")
    eval_parser.add_argument("--manifest-dir", type=Path, default=None)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--output", type=Path, default=None)
    eval_parser.add_argument("--overwrite", action="store_true")
    eval_parser.set_defaults(handler=cmd_evaluate)

    predict_parser = subparsers.add_parser("predict", help="classify one text with a trusted checkpoint")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--top-k", type=int, default=3)
    predict_parser.set_defaults(handler=cmd_predict)

    batch_parser = subparsers.add_parser("predict-file", help="classify texts from CSV or JSONL")
    batch_parser.add_argument("--checkpoint", type=Path, required=True)
    batch_parser.add_argument("--input", type=Path, required=True)
    batch_parser.add_argument("--output", type=Path, required=True)
    batch_parser.add_argument("--device", default="cpu")
    batch_parser.add_argument("--top-k", type=int, default=3)
    batch_parser.add_argument("--overwrite", action="store_true")
    batch_parser.set_defaults(handler=cmd_predict_file)

    export_parser = subparsers.add_parser("export-inference", help="export trusted .pt to safe inference files")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--overwrite", action="store_true")
    export_parser.set_defaults(handler=cmd_export_inference)

    compare_parser = subparsers.add_parser("compare-runs", help="compare compatible training runs")
    compare_parser.add_argument("runs", nargs="+", type=Path)
    compare_parser.add_argument("--metric", default="valid_macro_f1")
    compare_parser.add_argument("--output", type=Path, default=None)
    compare_parser.set_defaults(handler=cmd_compare)

    config_parser = subparsers.add_parser("show-config", help="print and validate resolved configuration")
    _config_parser(config_parser)
    config_parser.set_defaults(handler=lambda args: print(show_config(load_config(args.config, args.overrides))))
    models_parser = subparsers.add_parser("list-models", help="list built-in models")
    models_parser.set_defaults(handler=lambda args: print("\n".join(list_models())))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main(sys.argv[1:])
