# 教程

[English](README.md) | [文档首页](../README.zh-CN.md)

教程围绕同一条 AG News 链路展开，建议按顺序阅读和执行：

1. [文本分类基础](00-basics.zh-CN.md)：样本、标签、logits、loss 与划分。
2. [环境与安装](01-environment.zh-CN.md)：uv、CPU 检查和 Kaggle GPU 边界。
3. [数据与 token](02-data-and-tokens.zh-CN.md)：CSV、manifest、词表、截断和 padding。
4. [三个模型](03-models.zh-CN.md)：平均池化、卷积和双向 LSTM。
5. [训练与续训](04-training.zh-CN.md)：dry run、正常训练、选择 best 和恢复 last。
6. [评估与推理](05-evaluation-and-inference.zh-CN.md)：测试协议、错误样本和单文本预测。

第一次执行完整 CPU 小样本链路约需要以下命令：

```bash
uv sync --locked --extra dev
uv run python scripts/download_data.py --data-dir data/raw
uv run text-classify prepare-data --config configs/learning_minimal.yaml
uv run text-classify inspect-data --config configs/learning_minimal.yaml
uv run text-classify train --config configs/learning_minimal.yaml --dry-run --set device=cpu
uv run text-classify train --config configs/learning_minimal.yaml --set device=cpu --set run_name=tutorial-run
```

教程中的指标用于验证流程，不代表完整 AG News 性能。已完成的全量运行见[参考运行](../recorded-run/kaggle-agnews-textcnn/README.zh-CN.md)。
