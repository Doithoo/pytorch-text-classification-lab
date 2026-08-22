# Kaggle 训练指南

本项目以 Kaggle GPU 作为主要训练环境。推荐使用 Kaggle T4，项目按单 GPU 设计，使用 `cuda:0`，不依赖多卡。

## 准备

先将仓库推送到 GitHub，并修改 `docs/recorded-run/kaggle/run_kaggle.py` 中的 `PROJECT_URL`。然后安装 Kaggle CLI：

```bash
uv tool install kaggle
kaggle auth login
```

修改 `docs/recorded-run/kaggle/kernel-metadata.json` 中的：

```json
{
  "id": "你的用户名/pytorch-text-classification-lab-ag-news-gpu",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4"
}
```

`enable_internet` 必须开启，因为 runner 会下载项目代码和 AG News。`dataset_sources` 保持为空，避免非交互 Kernel 运行时动态挂载数据集失败。

## 提交

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <你的用户名>/pytorch-text-classification-lab-ag-news-gpu
```

runner 执行顺序为：

```text
git clone -> pip install -> download AG News -> prepare manifest -> inspect -> dry run -> CUDA train -> test evaluate
```

提交前确认仓库默认分支和 `PROJECT_URL` 可被 Kaggle 访问。第一次运行建议使用私有 Kernel。

## 产物

Kaggle 的 `/kaggle/working` 是可写临时磁盘。任务完成后立即下载：

```bash
kaggle kernels output <你的用户名>/pytorch-text-classification-lab-ag-news-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

重点文件：

```text
artifacts/kaggle-agnews-textcnn/best.pt
artifacts/kaggle-agnews-textcnn/last.pt
artifacts/kaggle-agnews-textcnn/metrics.csv
artifacts/kaggle-agnews-textcnn/evaluation/metrics.json
artifacts/kaggle-agnews-textcnn/evaluation/errors.jsonl
artifacts/kaggle-run-summary.json
```

不要只保存 `best.pt`。配置、metrics、evaluation 和运行摘要是复现实验结论所需的证据。

## 会话中断与续训

Kaggle 会话存在时间限制。项目保存 `last.pt`，可以在新的运行中将上一轮产物放入 `/kaggle/working/artifacts`，再使用：

```bash
text-classify train --config configs/reference_textcnn.yaml \
  --resume artifacts/kaggle-agnews-textcnn/last.pt \
  --set device=cuda
```

续训前应确认 tokenizer、manifest、模型配置和训练超参数没有发生不受控变化。

## 常见问题

| 问题 | 检查方式 |
|---|---|
| `torch.cuda.is_available()` 为 False | Kaggle Settings 是否选择 GPU |
| 找不到项目 | 检查 `PROJECT_URL` 是否公开可访问 |
| 下载失败 | 确认 Kernel 开启 Internet |
| OOM | 减小 batch size 或 max length |
| GPU 利用率低 | 增加 batch size、`num_workers` 或使用 TextCNN |
| 结果会话后消失 | 任务结束前或结束后下载 `artifacts/` |
| 续训结果异常 | 检查 manifest identity 和 tokenizer metadata |

参考运行发布前，应记录 Kaggle GPU 型号、PyTorch/CUDA 版本、Git revision、数据来源 hash、manifest hash、配置和完整产物。
