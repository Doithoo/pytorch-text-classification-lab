# 源码结构

[English](README.md) | [代码导览](../docs/concepts/code-tour.zh-CN.md)

可安装包是 `text_classifier`：

```text
config.py               配置合并与校验
cli.py                  text-classify 命令入口
data/                    AG News/generic CSV adapter、manifest、审计、tokenizer、Dataset
models/                  三个分类器和注册表
training/                训练循环、checkpoint、运行 metadata
evaluation/              分类指标和兼容运行比较
inference/               可信 checkpoint 单文本与文件批量预测
```

模块间依赖方向是 CLI -> data/model/training/evaluation/inference；底层模块不应导入 CLI。训练和推理共享模型、tokenizer 和 checkpoint 协议，不复制反序列化逻辑。

公共 Python API 目前保持较小，主要用户界面是 CLI。重命名 registry 名称、配置字段、manifest 列或 checkpoint 字段都属于兼容性变化，需要测试和文档迁移说明。
