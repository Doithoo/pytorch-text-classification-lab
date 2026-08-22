# Evaluation and Inference

[中文](05-evaluation-and-inference.zh-CN.md) | [Tutorial index](README.md)

After model selection, evaluate `best.pt` on validation or test:

```bash
uv run text-classify evaluate --checkpoint artifacts/tutorial-run/best.pt \
  --manifest-dir data/manifests --split test --device cpu
```

The default destination is `artifacts/tutorial-run/evaluation/test/`, containing `metrics.json` and confidence-sorted `errors.jsonl`. Validation uses a separate `evaluation/valid/` directory. Existing evidence is not overwritten unless `--overwrite` is explicit.

Evaluation checks the current `dataset.json` identity and label order against the checkpoint. This prevents silent evaluation on a different split with the same path shape. Error rows contain numeric IDs and label names, making pairs such as `business -> sci_tech` easy to filter.

Single-text prediction does not need manifests:

```bash
uv run text-classify predict --checkpoint artifacts/tutorial-run/best.pt \
  --text "The team won the championship final." --top-k 3 --device cpu
```

Output includes the top label, confidence, top-k entries, and every class probability. TextCNN safely handles empty and very short input. The executable example uses the same inference API:

```bash
uv run python examples/05_checkpoint_prediction.py \
  --checkpoint artifacts/tutorial-run/best.pt --text "Oil prices rose on Monday."
```

PyTorch checkpoints use pickle. Never load an untrusted `.pt` file. A softmax score is not necessarily calibrated probability or business risk; inspect the confusion matrix and high-confidence errors before interpreting it.
