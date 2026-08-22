# 数据格式与 Identity

[English](dataset-format.md) | [文档首页](../README.zh-CN.md)

原始 AG News 文件位于：

```text
<data_dir>/ag_news_csv/train.csv
<data_dir>/ag_news_csv/test.csv
```

每行由 CSV 解析器读取为 `label,title,description`。引号、逗号和换行必须符合标准 CSV，不能用简单字符串切分。

准备后的三个 manifest 使用相同 schema：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | string | `train-000000` 或 `test-000000` 稳定源行 ID |
| `text` | string | title 与 description 拼接文本 |
| `label` | string | world、sports、business、sci_tech |
| `label_id` | integer | 按上面顺序的 0-3 |

`dataset.json` 的 `schema_version=1`，保存 dataset 名称、标签顺序、seed、valid ratio、各 split 数量、源文件 SHA-256 和 manifest SHA-256。manifest identity 是该 JSON 规范化排序后内容的 SHA-256；checkpoint 保存它并在续训和评估时比较。

当前划分按类别分别 shuffle 和抽取验证集，再按稳定 ID 排序。相同源文件、seed 和 valid ratio 会得到相同 manifest。修改任何源内容、划分参数或标签协议都应产生新的 identity。
