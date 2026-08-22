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
    run(sys.executable, "-c", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))")
    run(sys.executable, "scripts/download_data.py", "--data-dir", "/kaggle/working/data/raw")
    run("text-classify", "prepare-data", "--config", "configs/reference_textcnn.yaml", "--set", "data.data_dir=/kaggle/working/data/raw", "--set", "data.manifest_dir=/kaggle/working/data/manifests")
    run("text-classify", "inspect-data", "--config", "configs/reference_textcnn.yaml", "--set", "data.manifest_dir=/kaggle/working/data/manifests")
    run("text-classify", "train", "--config", "configs/learning_minimal.yaml", "--dry-run", "--set", "data.data_dir=/kaggle/working/data/raw", "--set", "data.manifest_dir=/kaggle/working/data/manifests", "--set", "device=cuda")
    run("text-classify", "train", "--config", "configs/reference_textcnn.yaml", "--set", "data.data_dir=/kaggle/working/data/raw", "--set", "data.manifest_dir=/kaggle/working/data/manifests", "--set", "output_dir=/kaggle/working/artifacts", "--set", "device=cuda")
    checkpoint = "/kaggle/working/artifacts/kaggle-agnews-textcnn/best.pt"
    run("text-classify", "evaluate", "--checkpoint", checkpoint, "--split", "test", "--device", "cuda", "--output", "/kaggle/working/artifacts/kaggle-agnews-textcnn/evaluation")
    summary = {
        "project_url": PROJECT_URL,
        "git_revision": revision,
        "elapsed_seconds": round(time.time() - started, 2),
        "device": "cuda",
        "checkpoint": checkpoint,
    }
    Path("/kaggle/working/artifacts/kaggle-run-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "completed", **summary}), flush=True)


if __name__ == "__main__":
    main()
