# 数据与 Token

[English](02-data-and-tokens.md) | [教程首页](README.zh-CN.md)

下载脚本从固定 URL 获取 AG News CSV，并在写入前校验 SHA-256：

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run text-classify prepare-data --config configs/learning_minimal.yaml
uv run text-classify inspect-data --config configs/learning_minimal.yaml
```

源 CSV 的每行是 `label,title,description`。准备阶段把标题和描述拼接，标签从 1-4 转为 0-3，并按类别分层抽取验证集。输出的 `train.csv`、`valid.csv`、`test.csv` 字段固定为 `id,text,label,label_id`；`dataset.json` 保存标签顺序、样本数、随机种子、源哈希和 manifest 哈希。

`SimpleWordTokenizer` 使用 Unicode 正则、`casefold()` 和四个特殊 token：`<pad>`、`<unk>`、`<bos>`、`<eos>`。词频只从训练 manifest 统计，避免验证或测试词汇泄漏。序列超过 `max_length` 时保留 BOS/EOS 并截断正文。

同一 batch 内按最长样本动态 padding。`attention_mask=1` 表示真实 token，`0` 表示 padding。TextCNN 还会在序列短于最大卷积核时安全补齐，并屏蔽超出每个样本有效长度的卷积窗口。

先运行 `inspect-data` 查看长度分位数、截断数量和标签分布。若大量文本被截断，再在显存和速度允许范围内调整 `max_length`。数据文件协议见[数据格式参考](../reference/dataset-format.zh-CN.md)。
