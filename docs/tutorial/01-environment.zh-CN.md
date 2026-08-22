# 环境与安装

[English](01-environment.md) | [教程首页](README.zh-CN.md)

项目支持 Python 3.10-3.12，并使用 `uv.lock` 固定直接和间接依赖。推荐从全新克隆开始：

```bash
uv sync --locked --extra dev
uv run text-classify --version
uv run text-classify list-models
uv run pytest
```

这些命令不下载数据或模型权重。`uv run` 会使用项目虚拟环境，避免全局 Python 和项目依赖混用。

本地 CPU 用于示例、测试和小样本训练。`device=auto` 按 CUDA、Apple MPS、CPU 的顺序选择可用设备；需要确认环境时显式设置 `--set device=cpu`。请求不可用的 CUDA 或 MPS 会直接报错。

完整 AG News 参考训练以 Kaggle T4 为主。Kaggle runner 在线克隆仓库和下载数据，因此 Kernel 必须开启 Internet 和 GPU。具体步骤见[Kaggle 指南](../guides/kaggle.zh-CN.md)。

开发环境额外包含 Ruff、mypy、pytest、build、twine 和 pre-commit。绘图依赖是可选项：

```bash
uv run --extra plot python scripts/generate_doc_assets.py \
  --run-dir docs/recorded-run/kaggle-agnews-textcnn
```
