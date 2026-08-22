# AG News TextCNN Kaggle 参考运行

这是项目第一次真实 Kaggle GPU 运行，由 `docs/recorded-run/kaggle/run_kaggle.py` 提交，使用 Git revision：

```text
4ed733b532b9169816d02797fece33e34135bc20
```

| 项目 | 结果 |
| --- | ---: |
| 数据集 | AG News |
| 训练 / 验证 / 测试 | 108,000 / 12,000 / 7,600 |
| 模型 | TextCNN，embedding 256，channels 128，卷积核 3/4/5 |
| 设备 | Kaggle Tesla T4 |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 |
| Python | 3.12.13 |
| Epoch | 8 |
| 最佳 epoch | 3 |
| 最佳验证 macro-F1 | **0.915909** |
| 测试 accuracy | **0.914605** |
| 测试 macro-F1 | **0.914610** |
| 运行时间 | 151.3 秒 |

本次运行使用的数据源 hash：

```text
train.csv  76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e
test.csv   521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699
```

manifest identity：

```text
46780f09619b8d57e203a2719b2eec3ec3ea2b2a4039619a0673a1488ebb3447
```

仓库保留配置、tokenizer、训练曲线、测试指标、错误样本和 Kaggle 运行摘要。93 MB 的 checkpoint 不提交到 Git，仍可从 Kaggle Kernel 输出下载。逐类别指标和混淆矩阵位于 `evaluation/metrics.json`。

Kaggle Notebook：https://www.kaggle.com/code/yashowhoo/pytorch-text-classification-lab-ag-news-gpu
