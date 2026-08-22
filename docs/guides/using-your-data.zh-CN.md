# 使用自己的数据

[English](using-your-data.md) | [文档首页](../README.zh-CN.md)

`generic_csv` adapter 支持任意二分类或多分类任务。准备两个带表头的 UTF-8 CSV：

```text
data/custom/train.csv
data/custom/test.csv
```

默认字段是 `text,label`：

```csv
text,label
The team won the final,sports
Shares rose after earnings,business
```

训练集每个标签至少需要两条样本，项目才能按类抽取验证集。测试集标签必须在训练标签中出现。标签名称按训练集中去重后稳定排序，这个顺序写入 `dataset.json` 和 checkpoint。

```bash
uv run text-classify prepare-data --config configs/generic_csv_example.yaml
uv run text-classify inspect-data --config configs/generic_csv_example.yaml
uv run text-classify train --config configs/generic_csv_example.yaml --dry-run --set device=cpu
```

字段名不同可以覆盖：

```bash
uv run text-classify prepare-data --config configs/generic_csv_example.yaml \
  --set data.text_column=body --set data.label_column=category
```

`inspect-data` 除终端摘要外还写入 `inspection.json`，包含各 split 标签数、重复 ID、split 内重复文本、跨 split 文本泄漏、冲突标签、长度与截断、词表大小和 OOV 比例。发现跨 split 重复或冲突标签时应先修数据，再比较模型。

当前 adapter 要求用户提供 train/test，并从 train 生成 valid；尚不支持多标签分类、无标签预测文件、层级标签或流式数据。原始和私有文本不应提交仓库。错误分析与批量预测会保留正文，发布前检查隐私、版权和许可。
