# 数据格式与 Identity

[English](dataset-format.md) | [文档首页](../README.zh-CN.md)

项目有两个原始数据 adapter，但产出同一个 manifest 协议。

AG News 位于 `<data_dir>/ag_news_csv/{train,test}.csv`，每行是无表头 `label,title,description`。`generic_csv` 位于 `<data_dir>/{train,test}.csv`，要求表头；正文和标签字段默认 `text,label`，可由配置修改。

准备后的三个 manifest 字段固定：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | string | adapter 生成的稳定 split 行 ID |
| `text` | string | 模型输入正文 |
| `label` | string | 原始标签名称 |
| `label_id` | integer | `dataset.json.labels` 中的索引 |

`dataset.json` 使用 `schema_version=1`，保存 dataset/adapter、通用 CSV 列名、标签顺序、seed、valid ratio、split 数量、源 SHA-256 和 manifest SHA-256。manifest identity 是该 metadata 按键排序序列化后的 SHA-256。

通用 CSV 标签从训练集去重并按字符串排序。训练每类至少两条，测试不能出现训练外标签。valid 从训练集按类随机抽取后恢复稳定 ID 顺序。

`inspection.json` 不是 identity 的组成部分，而是派生审计证据，记录重复 ID、split 内重复、跨 split 归一化文本交集、冲突标签、长度/截断和 tokenizer OOV。修改源内容、adapter、列名、标签或划分参数会改变 identity；修改审计展示不会。
