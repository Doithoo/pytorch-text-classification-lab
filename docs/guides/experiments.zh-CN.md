# 实验指南

[English](experiments.md) | [文档首页](../README.zh-CN.md)

一个可比较实验必须先写清问题，例如“在相同词表和训练预算下，TextCNN 是否比 EmbeddingBag 提高验证 macro-F1”。每次只改变一个主要因素，并使用不同 `run_name`：

```bash
uv run text-classify train --config configs/learning_minimal.yaml \
  --set run_name=baseline --set model.name=embedding_bag
uv run text-classify train --config configs/learning_minimal.yaml \
  --set run_name=textcnn --set model.name=text_cnn
```

在运行前保存 `show-config` 输出，确认 manifest、seed、数据上限、epoch、batch size、学习率和 optimizer 一致。上例只适合流程演示；正式比较应确认 TextCNN kernel 与 max length 合法，并使用足够数据。

比较 `metrics.csv` 的最佳验证指标和 `run.json` 的耗时、代码 revision、配置/tokenizer/manifest 哈希。模型选择完成后，才分别评估同一个 test manifest。报告负结果和异常运行，不要只保留最高数字。

当前没有自动聚合命令，避免不同协议被静默合并。可使用 pandas 或电子表格显式整理；未来增加 `compare-runs` 时必须先验证 identity 和指标 schema。
