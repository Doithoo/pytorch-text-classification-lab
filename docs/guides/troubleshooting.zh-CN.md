# 排错指南

[English](troubleshooting.md) | [文档首页](../README.zh-CN.md)

| 现象 | 原因与处理 |
| --- | --- |
| 缺少 `data/manifests/train.csv` | 先下载数据，再运行 `prepare-data` |
| `unrecognized arguments: --data-dir` | CLI 数据路径通过 `--config` 或 `--set data.data_dir=...` 设置 |
| checksum mismatch | 文件不完整或来源变化；删除错误文件后重试，不要跳过校验 |
| `CUDA was requested but is not available` | 改用 CPU，或在 Kaggle Settings 开启 GPU |
| TextCNN kernel 超过 max length | 减小 `model.kernel_sizes` 或增大 `data.max_length` |
| 运行目录已存在 | 使用新的 `run_name`，不要覆盖已有实验 |
| evaluation 已存在 | 检查旧结果；确需替换时使用 `--overwrite` |
| manifest identity 不匹配 | checkpoint 和当前划分不是同一数据证据，重新准备正确 manifest |
| resume 配置不兼容 | 保持模型、tokenizer、优化器和学习率不变，只增加总 epoch |
| `best.pt` 不存在 | 检查 epoch 是否完成、验证集是否非空和日志错误 |
| MPS 算子问题 | 显式 `--set device=cpu` 复现，再报告 PyTorch 与 macOS 版本 |

开启问题前运行：

```bash
uv run text-classify show-config --config <配置路径>
uv run ruff check .
uv run pytest
```

问题报告应包含完整命令、错误栈、操作系统、Python/PyTorch 版本和最小配置。不要上传私有文本、Kaggle token 或来源不明的 checkpoint。
