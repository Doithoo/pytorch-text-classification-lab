# Training and Resume

[中文](04-training.zh-CN.md) | [Tutorial index](README.md)

Inspect the merged and validated configuration before training:

```bash
uv run text-classify show-config --config configs/learning_minimal.yaml \
  --set train.optimizer=adamw --set device=cpu
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
```

Precedence is defaults, YAML, then repeated `--set KEY=VALUE`. Unknown fields and invalid ranges fail during loading. A dry run builds the model and performs one forward, cross-entropy, and backward batch. It writes no checkpoint and does not indicate convergence.

A normal run needs a unique name:

```bash
uv run text-classify train --config configs/learning_minimal.yaml \
  --set device=cpu --set run_name=tutorial-run
```

Each epoch records training loss and validation metrics. `best.pt` changes only when `best_metric` improves; `last.pt` changes every epoch. The test split is absent from the training loop. `metrics.csv`, `config.yaml`, `tokenizer.json`, and `run.json` are evidence that belongs with checkpoints.

Resume a trusted `last.pt` after increasing total epochs:

```bash
uv run text-classify train --config configs/reference_textcnn.yaml \
  --resume artifacts/kaggle-agnews-textcnn/last.pt --set train.epochs=12
```

Resume requires an unchanged manifest, model, tokenizer settings, optimizer, learning rate, and best metric. Total epochs, device, and batch size may change. Optimizer and AMP scaler states are restored. `deterministic=true` requests deterministic PyTorch algorithms and may reduce performance or reject unsupported operations. The default fixes common seeds but does not promise bitwise identity across hardware.
