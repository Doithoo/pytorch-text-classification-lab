from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_classifier.inference import predict_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one text with a trusted project checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", default="Stocks rose after the company reported strong quarterly earnings.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    result = predict_text(args.checkpoint, args.text, args.device, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
