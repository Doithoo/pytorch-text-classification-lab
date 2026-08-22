# 添加模型

[English](adding-models.md) | [文档首页](../README.zh-CN.md)

内置模型必须遵守统一协议：构造时接收词表大小和类别数，`forward(input_ids, attention_mask)` 接收两个 `[B,L]` 长整型张量并返回 `[B,C]` logits。模型不得在 forward 内应用 softmax。

添加模型需要同步修改：

1. 在 `src/text_classifier/models/` 实现模块并处理 padding。
2. 在 `build_model()` 注册稳定名称，并在 `list_models()` 暴露。
3. 在配置校验中声明允许字段和范围。
4. 添加形状、短序列、padding、反向传播和 checkpoint round-trip 测试。
5. 更新 model zoo、配置参考、示例配置和中英文文档。

若模型需要完全不同的 tokenizer 或 batch 字段，不应把条件分支塞进现有 classifier。先提出新的数据/模型契约和 checkpoint schema 兼容方案。

预训练 Transformer 还需要模型与 tokenizer revision、下载缓存、attention/token type 字段、权重许可、分层学习率和安全序列化。目前这些能力未实现，因此不能只添加一个依赖和模型名称就宣称支持。
