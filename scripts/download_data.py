from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

URLS = {
    "train.csv": (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e",
    ),
    "test.csv": (
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
        "521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AG News CSV files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = args.data_dir / "ag_news_csv"
    target.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_sha256) in URLS.items():
        output = target / name
        if output.exists() and not args.force:
            actual_sha256 = sha256(output)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"checksum mismatch for existing {output}: {actual_sha256}")
            print(f"exists: {output} sha256={actual_sha256}")
            continue
        temporary = output.with_suffix(".download")
        print(f"download: {url}")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual_sha256 = sha256(temporary)
        if actual_sha256 != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {name}: expected {expected_sha256}, got {actual_sha256}")
        temporary.replace(output)
        print(f"saved: {output} bytes={output.stat().st_size} sha256={actual_sha256}")


if __name__ == "__main__":
    main()
