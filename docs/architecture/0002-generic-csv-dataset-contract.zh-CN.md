# ADR 0002：通用 CSV 数据协议

[English](0002-generic-csv-dataset-contract.md) | [文档首页](../README.zh-CN.md)

状态：Accepted

## 背景

`0.2.0` 的模型和 checkpoint 已能使用 metadata 标签，但数据准备仍把 AG News 路径、无表头三列格式和四类标签写死，导致“类别数动态”不能转化为真实自有数据支持。

## 决策

保留 `ag_news` adapter 以兼容已有运行，新增 `generic_csv` adapter。它读取带表头的 train/test CSV，字段名可配置，从训练标签稳定排序生成 ID，并按类抽取 valid。两种 adapter 产出同一个 manifest schema 和 identity 协议。

数据审计作为准备后的独立证据：检查重复 ID、重复文本、跨 split 泄漏、冲突标签、长度、截断和 tokenizer OOV。训练仍只消费 manifest，不感知原始 adapter。

## 结果

任意二分类和多分类数据可以复用现有三个模型、checkpoint、评估和推理。代价是标签排序成为公共契约，改变字段、标签文本或 adapter 都会改变 identity。多标签、无标签测试和流式输入需要未来新 schema，不能隐式塞入当前协议。
