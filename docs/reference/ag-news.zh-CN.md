# AG News 数据集说明

[English](ag-news.md) | [文档首页](../README.zh-CN.md)

AG News 是新闻主题分类数据集。项目使用的 CSV 镜像来自 `mhjabreel/CharCnn_Keras`，包含 120,000 条训练源样本和 7,600 条测试样本，四类均衡。项目再从训练源按类抽取 10% 作为验证集。

原始论文引用：

```bibtex
@inproceedings{zhang2015character,
  title={Character-level Convolutional Networks for Text Classification},
  author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle={Advances in Neural Information Processing Systems},
  year={2015}
}
```

数据起源可追溯到 AG's corpus of news articles。镜像仓库和原始数据的许可说明不等同于本项目 MIT License。使用、发布或再分发原始文本、派生词表和错误样本前，应自行核对上游条款、所在地区法律和使用场景。

本仓库不提交原始 train/test CSV 或 checkpoint，但参考运行提交了派生 tokenizer 和包含测试文本的错误样本，用于实验审计。若组织策略不允许再分发文本，应只发布聚合指标、样本 ID 和哈希，删除正文。

下载脚本固定 URL 和 SHA-256，以检测上游内容变化；校验和证明字节身份，不授予数据使用权。
