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

本仓库不提交原始 train/test CSV、checkpoint、完整 tokenizer 或逐条错误正文。参考运行保留聚合证据；若组织策略允许，可从 Kaggle 输出获取详细错误和 tokenizer。发布派生词表或源文本前，应核对上游条款、隐私和版权。

下载脚本固定 URL 和 SHA-256，以检测上游内容变化；校验和证明字节身份，不授予数据使用权。
