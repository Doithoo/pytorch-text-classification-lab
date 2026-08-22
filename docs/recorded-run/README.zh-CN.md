# 参考运行索引

[English](README.md) | [文档首页](../README.zh-CN.md)

本目录包含可复现 Kaggle runner、Kernel metadata 和已完成运行的机器可读证据。大 checkpoint 不提交 Git，按运行页面中的 Kaggle 链接下载。

## 已发布运行

| 运行 | 模型 | Test accuracy | Test macro-F1 | 页面 |
| --- | --- | ---: | ---: | --- |
| `kaggle-agnews-textcnn` | TextCNN，旧 revision | 0.914605 | 0.914610 | [详情](kaggle-agnews-textcnn/README.zh-CN.md) |
| `kaggle-agnews-model-comparison-v0.3.0` | EmbeddingBag / TextCNN / BiLSTM | 0.915395 / 0.910132 / 0.916053 | 0.915278 / 0.910090 / **0.915985** | [详情](kaggle-agnews-model-comparison-v0.3.0/README.zh-CN.md) |

## Runner

`kaggle/` 是单模型 runner，`kaggle-comparison/` 是当前代码的三模型 runner。后者已经在 revision `0b350f2` 完成一次真实 T4 运行；再次实验应创建新的版本目录，不覆盖历史证据。

每个完整运行应保留精确 git revision、源与 manifest identity、最终配置、逐轮 metrics、GPU/PyTorch/CUDA/Python、聚合测试指标和运行摘要。逐条错误正文与 tokenizer 可作为本地/Kaggle 输出，不默认提交 Git。训练选择只使用验证集，测试集在选择完成后评估一次。
