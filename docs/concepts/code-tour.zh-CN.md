# 代码导览

[English](code-tour.md) | [文档首页](../README.zh-CN.md)

建议按数据流而不是目录字母顺序阅读：

1. `src/text_classifier/config.py`：默认值、YAML 合并、命令行覆盖和集中校验。
2. `data/manifest.py`：AG News 解析、分层划分和数据 identity。
3. `data/tokenizer.py`、`data/dataset.py`：token ID、截断、padding 和 mask。
4. `models/classifiers.py`：三个 `[B,L] -> [B,C]` 模型。
5. `training/train.py`：设备、优化器、epoch、验证、checkpoint 和运行元数据。
6. `training/checkpoint.py`：可信 checkpoint 的结构验证和旧配置默认值补全。
7. `inference/predictor.py`、`evaluation/metrics.py`：预测与分类指标。
8. `cli.py`：把上述模块组织成用户命令。

CLI 保持薄层：参数解析、路径决策和 JSON 输出在 CLI；模型训练、推理和数据准备留在可测试的 Python API 中。新增功能时应沿用这个边界。

`examples/` 只演示单一概念，不应复制完整训练实现。`scripts/` 处理下载和文档资产等仓库维护任务。`docs/recorded-run/` 保存可发布证据，不参与包运行时。
