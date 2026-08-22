from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _plot_training(metrics_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    loss = [float(row["train_loss"]) for row in rows]
    accuracy = [float(row["valid_accuracy"]) for row in rows]
    macro_f1 = [float(row["valid_macro_f1"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, loss, marker="o", color="#c2410c")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, accuracy, marker="o", label="Accuracy", color="#0369a1")
    axes[1].plot(epochs, macro_f1, marker="s", label="Macro-F1", color="#15803d")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0.88, 0.93))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_confusion(metrics_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    matrix = np.asarray(metrics["confusion"], dtype=float)
    labels = metrics.get("label_names", ["world", "sports", "business", "sci_tech"])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(
        title="AG News test confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = int(matrix[row, column])
            color = "white" if normalized[row, column] > 0.55 else "black"
            axis.text(column, row, f"{value}\n{normalized[row, column]:.1%}", ha="center", va="center", color=color)
    figure.colorbar(image, ax=axis, label="Row-normalized rate")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate documentation plots from a recorded run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    _plot_training(args.run_dir / "metrics.csv", args.output_dir / "ag-news-textcnn-training.png")
    _plot_confusion(
        args.run_dir / "evaluation" / "metrics.json",
        args.output_dir / "ag-news-textcnn-confusion.png",
    )
    print(f"wrote documentation assets to {args.output_dir}")


if __name__ == "__main__":
    main()
