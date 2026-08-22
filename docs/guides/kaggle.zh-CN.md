# Kaggle 训练指南

[English](kaggle.md) | [文档首页](../README.zh-CN.md)

项目的完整训练环境是单张 Kaggle Tesla T4。runner 使用 `cuda:0`，不依赖多卡。

## 准备

安装并认证 Kaggle CLI：

```bash
uv tool install kaggle
kaggle auth login
```

打开 `docs/recorded-run/kaggle/kernel-metadata.json`，把 `id` 中的 `yashowhoo` 改成你的 Kaggle 用户名。保留 GPU、Internet 和 T4 设置。runner 会在线克隆公开仓库并下载 AG News，因此默认分支必须可访问；若使用 fork，同步修改 `run_kaggle.py` 的 `PROJECT_URL`。

## 提交和观察

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <你的用户名>/pytorch-text-classification-lab-ag-news-gpu
```

执行顺序是：

```text
git clone -> install -> download -> prepare -> inspect -> dry run -> train -> evaluate
```

日志首先断言 CUDA 可用，再执行 CPU 最小配置 dry run，最后使用 `reference_textcnn.yaml` 完整训练。完整参考运行约 2.5 分钟，但排队、网络和不同 Kaggle 镜像会改变总时间。

## 下载证据

状态为 `COMPLETE` 后执行：

```bash
kaggle kernels output <你的用户名>/pytorch-text-classification-lab-ag-news-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

保留 `best.pt`、`last.pt`、`config.yaml`、`tokenizer.json`、`metrics.csv`、`run.json`、evaluation 和 `kaggle-run-summary.json`。Kaggle 的 `/kaggle/working` 是临时磁盘，不下载就会丢失。

## 中断和续训

把上一轮完整运行目录恢复到相同路径，并增加总 epoch：

```bash
text-classify train --config configs/reference_textcnn.yaml \
  --resume artifacts/kaggle-agnews-textcnn/last.pt \
  --set train.epochs=12 --set device=cuda
```

不要从不可信来源加载 checkpoint。续训会验证 manifest、模型、tokenizer 和优化器关键参数。

## 三模型对比

`docs/recorded-run/kaggle-comparison/` 提供当前代码的 EmbeddingBag、TextCNN、BiLSTM 同 manifest runner：

```bash
kaggle kernels push -p docs/recorded-run/kaggle-comparison
```

它会生成三个运行目录和经过 identity 校验的 `comparison.json`。该 runner 尚未完成真实运行，因此文档不声明新成绩。

## 发布结果

先用验证集选择设置，再评估一次测试集。发布页面应给出精确 git revision、数据和 manifest identity、最终配置、依赖环境、GPU、墙钟时间、完整指标和错误分析，不发布估算数字。
