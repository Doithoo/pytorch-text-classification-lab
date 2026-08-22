# 测试

[English](README.md) | [贡献指南](../CONTRIBUTING.zh-CN.md)

测试只使用合成 CSV、临时目录和小模型，不下载数据或权重。完整运行：

```bash
uv run pytest -W error::DeprecationWarning
```

| 测试 | 主要契约 |
| --- | --- |
| `test_config.py` | 合并、类型、范围、未知字段和路径安全 |
| `test_manifest.py` | AG News/generic CSV、固定分层划分、metadata、审计和非法比例 |
| `test_data.py` | 训练词表、padding 和 mask |
| `test_models.py` | 三模型形状、TextCNN 短文本和 padding 稳定性 |
| `test_checkpoint.py` | schema、可信加载、safetensors 导出和短文本预测 |
| `test_metrics.py` | 混淆矩阵、macro 指标和输入校验 |
| `test_training.py` | 优化器选择和设备失败路径 |
| `test_inference_and_compare.py` | 批量预测、运行 identity 校验和比较排序 |
| `test_end_to_end.py` | AG News 与通用三分类训练、续训、评估防覆盖和 CLI 预测 |
| `test_documentation.py` | 双语页面、链接、命令、配置、示例和模型清单 |
| `test_packaging.py` | 构建元数据和版本入口 |

新增行为应在最接近的层测试，并在跨模块契约变化时补端到端测试。不要依赖网络、GPU、用户主目录或已有 `data/`。
