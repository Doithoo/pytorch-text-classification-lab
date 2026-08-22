# ADR 0001：可复现文本分类契约

[English](0001-reproducible-text-classification-contracts.md) | [文档首页](../README.zh-CN.md)

状态：Accepted

## 背景

入门项目常把数据下载、随机划分、词表、训练和测试写在一个 Notebook 中。Notebook 容易运行，却难以确认两个结果是否使用相同数据、标签顺序和模型选择协议。

## 决策

项目采用固定的可审计边界：原始数据不提交；准备阶段生成带 SHA-256 的 manifest；词表只从训练集构建；配置在训练前完全解析并校验；训练只访问 train/valid；checkpoint 保存模型、优化器、tokenizer、标签和 manifest identity；测试评估写入独立且默认不可覆盖的目录；真实结果连同环境和错误样本发布。

初始版本只支持 AG News、简单词级 tokenizer 和三个经典模型。`0.3.0` 按 ADR 0002 增加通用 CSV adapter，但继续拒绝未实现的多标签和 Transformer 配置。

## 结果

实验更容易比较和审计，CLI 与文件协议也能由测试约束。代价是增加了 manifest、metadata 和文档维护工作；改变标签、checkpoint 或配置结构时需要显式兼容策略。

新增 adapter 或预训练模型时，应扩展显式 adapter/registry 和 schema，而不是在 AG News 分支中增加隐式条件。
