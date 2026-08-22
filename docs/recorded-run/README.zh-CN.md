# 参考运行证据

[English](README.md) | [文档首页](../README.zh-CN.md)

本目录包含可复现 Kaggle runner、Kernel metadata 和已完成运行的机器可读证据。

当前已发布运行：

| 运行 | 模型 | 测试 accuracy | 测试 macro-F1 | 页面 |
| --- | --- | ---: | ---: | --- |
| `kaggle-agnews-textcnn` | TextCNN | 0.914605 | 0.914610 | [详情](kaggle-agnews-textcnn/README.zh-CN.md) |

运行目录保留精确 git revision、解析配置、源与 manifest identity、tokenizer、逐轮 metrics、Kaggle GPU/PyTorch/CUDA/Python、测试指标和高置信度错误。93 MB checkpoint 由 Kaggle 输出提供，不提交 Git。

`kaggle/` 是可重新提交的单模型 runner，`kaggle-comparison/` 是尚未产生发布成绩的三模型 runner；两者都不是完成运行的证据。发布新结果时创建独立目录，不覆盖历史证据；不要把 dry run、小样本或待运行配置描述为完整基准。
