# 使用自己的数据

[English](using-your-data.md) | [文档首页](../README.zh-CN.md)

当前发布版只正式支持 AG News 的三列 CSV 和固定四类标签。把自己的文件改名为 `train.csv`、`test.csv` 并不等于兼容：解析器会把第一列按 1-4 解释，标签名称和预测输出也遵循 AG News 协议。

要在不修改源码的情况下做临时实验，自有数据必须满足：

```text
<label 1..4>,<title>,<description>
```

并放在 `<data_dir>/ag_news_csv/`。这只适合结构兼容的四分类实验；生成的 metadata 仍标记为 `ag_news`，因此不应发布为通用自定义数据支持。

正式增加数据集应实现独立 adapter，明确：数据源与许可、原始 schema、标签顺序、稳定样本 ID、train/valid/test 规则、重复样本处理、哈希 identity 和类别数。随后移除训练和推理中的 AG News 假设，并给 manifest metadata 增加 dataset adapter 名称与版本。

不要提交私有文本或原始数据。错误分析文件会保存完整文本，发布前必须检查隐私、版权和数据许可。
