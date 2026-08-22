# Checkpoint 协议

[English](checkpoint-schema.md) | [文档首页](../README.zh-CN.md)

项目 checkpoint 是 `torch.save` 生成的 mapping，当前 `schema_version=1`。必要字段包括：

| 字段 | 内容 |
| --- | --- |
| `model_name`、`model_config` | 模型注册名和构造参数 |
| `model_state_dict` | 模型参数 |
| `optimizer_state_dict` | 正常训练续训状态 |
| `scaler_state_dict` | AMP GradScaler 状态，可为空 |
| `epoch`、`metrics` | 已完成 epoch 索引和当轮验证指标 |
| `config` | 创建运行时的完整解析配置 |
| `tokenizer_metadata` | vocab、max length 和 tokenizer 名称 |
| `manifest_identity` | 训练数据协议 identity |
| `label_names` | 类别 ID 顺序；旧 v1 文件缺失时回退 AG News 标签 |

加载器会检查文件存在、mapping、schema、必要字段、配置和 tokenizer 基本结构。旧 v1 配置缺少后来新增的默认字段时会补全默认值。

`best.pt` 用于评估和推理；`last.pt` 用于恢复中断训练。不要只保存权重而丢失 config、tokenizer 和 manifest metadata。

`torch.load(weights_only=False)` 使用 pickle，恶意文件可能执行代码。只加载由自己或可信项目运行生成的 checkpoint，不把上传的任意 `.pt` 当普通数据处理。schema 校验发生在反序列化之后，不能防止 pickle 执行。
