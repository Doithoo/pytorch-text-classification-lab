# 训练与续训

[English](04-training.md) | [教程首页](README.zh-CN.md)

训练前先查看合并并校验后的配置：

```bash
uv run text-classify show-config --config configs/learning_minimal.yaml \
  --set train.optimizer=adamw --set device=cpu
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
```

配置优先级是默认值、YAML、重复出现的 `--set KEY=VALUE`。未知字段和非法范围会在加载时失败。dry run 执行一个 batch 的模型构建、前向、交叉熵和反向传播，不写 checkpoint，也不说明模型已经收敛。

正常训练需要唯一 `run_name`：

```bash
uv run text-classify train --config configs/learning_minimal.yaml \
  --set device=cpu --set run_name=tutorial-run
```

每轮计算训练 loss 和验证指标。`best.pt` 只在 `best_metric` 改善时更新，`last.pt` 每轮更新。测试集不参与训练循环。`metrics.csv`、`config.yaml`、`tokenizer.json` 和 `run.json` 与 checkpoint 一起构成运行证据。

从中断处继续时使用可信的 `last.pt`，并把总 epoch 调大：

```bash
uv run text-classify train --config configs/reference_textcnn.yaml \
  --resume artifacts/kaggle-agnews-textcnn/last.pt --set train.epochs=12
```

续训要求 manifest、模型、tokenizer 参数、优化器、学习率和最佳指标不变；允许增加总 epoch、改变设备和 batch size。checkpoint 中的优化器与 AMP scaler 状态会恢复。`deterministic=true` 会请求 PyTorch 确定性算法，可能降低性能或让不支持的算子报错；默认只固定常用随机种子，不承诺跨硬件位级一致。
