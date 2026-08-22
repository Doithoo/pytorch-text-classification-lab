# 配置流

[English](configuration-flow.md) | [文档首页](../README.zh-CN.md)

配置按以下顺序合并：

```text
DEFAULT_CONFIG -> YAML -> --set KEY=VALUE -> validate_config
```

后面的值覆盖前面的值。`--set` 可以重复，值使用 YAML 解析，因此 `true` 是布尔值、`null` 是空值、`[3,4,5]` 是列表：

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml \
  --set train.epochs=12 --set model.kernel_sizes='[2,3,4]'
```

`show-config` 打印的结果就是训练写入 `config.yaml` 的结构。未知字段不会被静默保留；非法类型、范围、模型名、优化器、设备和 TextCNN 卷积核长度会在读取数据前失败。

字段分为数据、模型、训练和运行四类。续训时并非所有字段都可修改：模型、tokenizer、manifest identity、优化器和学习率必须与 checkpoint 一致；总 epoch、设备和 batch size可以改变。完整字段见[配置参考](../reference/config-reference.zh-CN.md)。
