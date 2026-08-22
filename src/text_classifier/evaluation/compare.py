from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_run(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    metrics_path = run_dir / "metrics.csv"
    if not run_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"run directory must contain run.json and metrics.csv: {run_dir}")
    metadata = json.loads(run_path.read_text(encoding="utf-8"))
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    if not isinstance(metadata, dict) or not history:
        raise ValueError(f"run evidence is empty: {run_dir}")
    return {"run_dir": str(run_dir), "metadata": metadata, "history": history}


def compare_runs(run_dirs: list[str | Path], metric: str = "valid_macro_f1") -> list[dict[str, Any]]:
    """Compare runs only when their data and label protocols are compatible."""
    allowed_metrics = {"valid_accuracy", "valid_macro_f1", "valid_macro_precision", "valid_macro_recall"}
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of: {', '.join(sorted(allowed_metrics))}")
    if len(run_dirs) < 2:
        raise ValueError("compare_runs requires at least two run directories")
    loaded = [_load_run(Path(path)) for path in run_dirs]
    metadata = [item["metadata"] for item in loaded]
    for key in ("manifest_identity", "tokenizer_sha256", "label_names"):
        if any(item.get(key) in (None, "", []) for item in metadata):
            raise ValueError(f"runs are missing required {key}")
        values = {json.dumps(item[key], sort_keys=True) for item in metadata}
        if len(values) > 1:
            raise ValueError(f"runs have incompatible {key}")
    rows: list[dict[str, Any]] = []
    for item in loaded:
        candidates = [row for row in item["history"] if row.get(metric) not in (None, "")]
        if not candidates:
            raise ValueError(f"metric {metric!r} is absent from {item['run_dir']}")
        best = max(candidates, key=lambda row: float(row[metric]))
        rows.append(
            {
                "run_dir": item["run_dir"],
                "best_epoch": int(best["epoch"]),
                "best_metric": float(best[metric]),
                "metric": metric,
                "manifest_identity": metadata[0].get("manifest_identity"),
                "git_revision": item["metadata"].get("git_revision"),
                "elapsed_seconds": item["metadata"].get("elapsed_seconds"),
            }
        )
    return sorted(rows, key=lambda row: float(row["best_metric"]), reverse=True)
