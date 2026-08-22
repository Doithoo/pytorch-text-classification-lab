from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_URL = "https://github.com/Doithoo/pytorch-text-classification-lab.git"
PROJECT_DIR = Path("/kaggle/working/project")
ARTIFACTS = Path("/kaggle/working/artifacts")
MANIFESTS = Path("/kaggle/working/data/manifests")
RAW_DATA = Path("/kaggle/working/data/raw")
CONFIGS = (
    ("configs/reference_embedding_bag.yaml", "kaggle-agnews-embedding-bag"),
    ("configs/reference_textcnn.yaml", "kaggle-agnews-textcnn"),
    ("configs/reference_bilstm.yaml", "kaggle-agnews-bilstm"),
)


def run(*command: str) -> None:
    print(json.dumps({"phase": "command", "command": list(command)}), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    started = time.time()
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    run("git", "clone", "--depth", "1", PROJECT_URL, str(PROJECT_DIR))
    os.chdir(PROJECT_DIR)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(sys.executable, "-m", "pip", "install", "-q", "-e", ".")
    run(
        sys.executable,
        "-c",
        "import torch; assert torch.cuda.is_available(); "
        "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))",
    )
    run(sys.executable, "scripts/download_data.py", "--data-dir", str(RAW_DATA))
    run(
        "text-classify",
        "prepare-data",
        "--config",
        "configs/reference_textcnn.yaml",
        "--set",
        f"data.data_dir={RAW_DATA}",
        "--set",
        f"data.manifest_dir={MANIFESTS}",
    )
    run(
        "text-classify",
        "inspect-data",
        "--config",
        "configs/reference_textcnn.yaml",
        "--set",
        f"data.manifest_dir={MANIFESTS}",
    )
    for config, run_name in CONFIGS:
        run(
            "text-classify",
            "train",
            "--config",
            config,
            "--set",
            f"data.data_dir={RAW_DATA}",
            "--set",
            f"data.manifest_dir={MANIFESTS}",
            "--set",
            f"output_dir={ARTIFACTS}",
            "--set",
            "device=cuda",
        )
        run(
            "text-classify",
            "evaluate",
            "--checkpoint",
            str(ARTIFACTS / run_name / "best.pt"),
            "--manifest-dir",
            str(MANIFESTS),
            "--split",
            "test",
            "--device",
            "cuda",
        )
    run(
        "text-classify",
        "compare-runs",
        *(str(ARTIFACTS / run_name) for _, run_name in CONFIGS),
        "--output",
        str(ARTIFACTS / "comparison.json"),
    )
    summary = {
        "schema_version": 1,
        "project_url": PROJECT_URL,
        "git_revision": revision,
        "elapsed_seconds": round(time.time() - started, 2),
        "device": "cuda",
        "runs": [run_name for _, run_name in CONFIGS],
    }
    (ARTIFACTS / "kaggle-comparison-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "completed", **summary}), flush=True)


if __name__ == "__main__":
    main()
