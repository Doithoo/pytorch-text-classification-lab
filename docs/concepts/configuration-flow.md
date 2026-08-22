# Configuration Flow

[中文](configuration-flow.zh-CN.md) | [Documentation index](../README.md)

Configuration is merged in this order:

```text
DEFAULT_CONFIG -> YAML -> --set KEY=VALUE -> validate_config
```

Later values replace earlier values. `--set` may repeat and values are parsed as YAML, so `true` is Boolean, `null` is null, and `[3,4,5]` is a list:

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml \
  --set train.epochs=12 --set model.kernel_sizes='[2,3,4]'
```

`show-config` prints the same resolved shape written to a run's `config.yaml`. Unknown fields are not retained silently. Invalid types, ranges, model names, optimizers, devices, and TextCNN kernel lengths fail before data access.

Fields belong to data, model, training, or run policy. Resume does not allow every field to change: model, tokenizer, manifest identity, optimizer, and learning rate must match the checkpoint. Total epochs, device, and batch size may change. See the [config reference](../reference/config-reference.md).
