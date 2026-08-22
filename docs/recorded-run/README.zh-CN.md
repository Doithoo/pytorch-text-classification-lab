# Kaggle 参考运行

本目录保存可复现的 Kaggle runner 和 Kernel metadata，目前不声明参考指标。需要先将 GitHub 仓库发布并完成第一次 T4 训练，再把真实结果写入这里。

发布的参考运行应保留：

- 精确 Git revision 和最终配置；
- 原始数据与 manifest 的 SHA-256 identity；
- Kaggle GPU、PyTorch、CUDA 和 Python 版本；
- `metrics.csv`、`best.pt`、`last.pt`、测试指标和高置信度错误样本；
- 总运行时间和未修改的 `kaggle-run-summary.json`。

第一次运行按 [Kaggle 训练指南](../guides/kaggle.zh-CN.md)操作。没有完成真实训练前，不应填写估算性能数字。
