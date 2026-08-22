# AG News TextCNN Kaggle 参考运行

[English](README.md) | [参考运行索引](../README.zh-CN.md)

这是项目第一次真实 Kaggle GPU 运行，由 `docs/recorded-run/kaggle/run_kaggle.py` 提交，精确代码 revision 为：

```text
4ed733b532b9169816d02797fece33e34135bc20
```

| 项目 | 结果 |
| --- | ---: |
| 数据集 | AG News |
| 训练 / 验证 / 测试 | 108,000 / 12,000 / 7,600 |
| 模型 | TextCNN，embedding 256，channels 128，卷积核 3/4/5 |
| 设备 | Kaggle Tesla T4 |
| PyTorch / CUDA / Python | 2.10.0+cu128 / 12.8 / 3.12.13 |
| Epoch | 8 |
| 最佳 epoch | 3 |
| 最佳验证 macro-F1 | **0.915909** |
| 测试 accuracy | **0.914605** |
| 测试 macro-F1 | **0.914610** |
| Kaggle 总耗时 | 151.3 秒 |

![训练与验证曲线](../../assets/ag-news-textcnn-training.png)

最佳轮次之后训练 loss 继续下降，但验证指标回落，说明后续 epoch 已过拟合；`best.pt` 而不是 `last.pt` 用于测试。

![测试混淆矩阵](../../assets/ag-news-textcnn-confusion.png)

649 条测试错误中，最大的双向混淆是 business -> sci_tech 155 条和 sci_tech -> business 129 条；186 条错误的预测置信度不低于 0.9。高置信度错误中存在标签边界模糊的体育人物新闻和国际事件，说明错误分析不能只归因于模型容量。

数据源 SHA-256：

```text
train.csv  76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e
test.csv   521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699
```

manifest identity：

```text
46780f09619b8d57e203a2719b2eec3ec3ea2b2a4039619a0673a1488ebb3447
```

仓库保留 `config.yaml`、`tokenizer.json`、`metrics.csv`、`run.json`、测试指标、错误样本和 Kaggle 摘要。93 MB 的 `best.pt`/`last.pt` 从 Kaggle 输出下载，不提交 Git。该结果严格属于上面的旧 revision；当前代码增加了 TextCNN padding mask、配置校验和新元数据，重新运行可能产生不同数字，不应静默替换本记录。

Kaggle Notebook：https://www.kaggle.com/code/yashowhoo/pytorch-text-classification-lab-ag-news-gpu
