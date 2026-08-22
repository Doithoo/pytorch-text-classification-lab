# AG News 三模型对比参考运行

[English](README.md) | [参考运行索引](../README.zh-CN.md)

这是当前 `0.3.0` 代码的第一次完整三模型 Kaggle 对比，由 `docs/recorded-run/kaggle-comparison/run_kaggle_comparison.py` 提交，使用精确 Git revision：

```text
0b350f27f3af6d1cbb417f064f3c546a9af11b46
```

三次训练使用同一份 AG News manifest、同一 tokenizer、同一随机种子、8 个 epoch 和 T4 GPU。模型选择只看 validation macro-F1，随后使用各自 `best.pt` 评估同一 test split。

| 模型 | 最佳 epoch | 最佳 valid macro-F1 | Test accuracy | Test macro-F1 | 训练耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BiLSTM | 4 | **0.924121** | **0.916053** | **0.915985** | 229.9 秒 |
| EmbeddingBag | 6 | 0.920197 | 0.915395 | 0.915278 | 61.4 秒 |
| TextCNN | 5 | 0.915575 | 0.910132 | 0.910090 | 159.0 秒 |

Kaggle runner 总耗时为 514.6 秒，设备为 Tesla T4，PyTorch `2.10.0+cu128`，CUDA `12.8`，Python `3.12.13`。

## 数据和协议

```text
train / valid / test = 108,000 / 12,000 / 7,600
manifest_identity = 46780f09619b8d57e203a2719b2eec3ec3ea2b2a4039619a0673a1488ebb3447
tokenizer_sha256 = 1797a6c4c25c25ba9dfdf881e32b3baf63ac59d72ebca5f1d160d9169985bb39
```

源文件 SHA-256 与旧 AG News 参考运行一致：

```text
train.csv  76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e
test.csv   521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699
```

测试错误数量分别为 BiLSTM 638、EmbeddingBag 643、TextCNN 683。逐类别指标、混淆矩阵和错误样本在每个模型目录的 `evaluation/test/` 中；完整配置、逐轮 metrics、tokenizer 和 run metadata 也一并保留。93 MB 级 checkpoint 不提交 Git，可从 Kaggle Kernel 输出下载。

## 解释边界

在相同数据和词表下，BiLSTM 得到最高测试 macro-F1，但训练耗时约为 EmbeddingBag 的 3.7 倍。EmbeddingBag 的结果接近 BiLSTM，说明 AG News 的词汇和局部主题线索对简单基线已经很有价值。当前 TextCNN 结果低于旧 revision 的 `0.914610`，不能直接归因于单一代码变化；当前比较使用了新的 masking 行为和独立配置，应以本记录为准，并保留旧记录作为历史证据。

这不是跨数据集或跨硬件的通用基准。模型选择、类别均衡、AG News 标签边界和单次随机种子都会影响结果。

Kaggle Kernel：https://www.kaggle.com/code/yashowhoo/pytorch-text-classification-lab-model-comparison
