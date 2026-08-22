from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import DEFAULT_CONFIG, _merge, validate_config
from ..training.checkpoint import load_checkpoint

SAFE_INFERENCE_SCHEMA_VERSION = 1


def _metadata_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(weights_path.suffix + ".json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_inference_checkpoint(
    checkpoint_path: str | Path, output_path: str | Path, overwrite: bool = False
) -> tuple[Path, Path]:
    """Convert a trusted training checkpoint into safe inference weights and JSON metadata."""
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("install the safe extra to export safetensors") from exc
    output = Path(output_path)
    if output.suffix != ".safetensors":
        raise ValueError("safe inference output must use the .safetensors suffix")
    metadata_path = _metadata_path(output)
    if not overwrite and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"safe inference output already exists: {output}")
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = {name: tensor.detach().cpu().contiguous() for name, tensor in payload["model_state_dict"].items()}
    metadata: dict[str, Any] = {
        "schema_version": SAFE_INFERENCE_SCHEMA_VERSION,
        "model_name": payload["model_name"],
        "config": payload["config"],
        "tokenizer_metadata": payload["tokenizer_metadata"],
        "label_names": payload["label_names"],
        "manifest_identity": payload["manifest_identity"],
        "source_checkpoint_schema_version": payload["schema_version"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output)
    metadata["weights_sha256"] = _sha256(output)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return output, metadata_path


def load_inference_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("install the safe extra to load safetensors") from exc
    weights_path = Path(path)
    metadata_path = _metadata_path(weights_path)
    if not weights_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"expected {weights_path} and {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SAFE_INFERENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported safe inference metadata: {metadata_path}")
    if metadata.get("weights_sha256") != _sha256(weights_path):
        raise ValueError("safe inference weights SHA-256 does not match metadata")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("safe inference config must be a mapping")
    metadata["config"] = validate_config(_merge(DEFAULT_CONFIG, config))
    labels = metadata.get("label_names")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("safe inference label_names must be a non-empty string list")
    tokenizer = metadata.get("tokenizer_metadata")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("vocab"), dict):
        raise ValueError("safe inference tokenizer metadata is invalid")
    metadata["model_state_dict"] = load_file(weights_path, device=device)
    return metadata
