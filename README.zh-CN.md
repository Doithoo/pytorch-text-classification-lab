# PyTorch 文本分类实践

这是一个以 Kaggle GPU 为主线的 PyTorch 文本分类学习项目。第一版使用 AG News，完整覆盖：

```text
下载数据 -> 生成固定 manifest -> 检查文本和长度 -> tokenizer -> padding batch -> dry run -> GPU 训练 -> 评估 -> 错误分析
```

项目不要求本地 CUDA。Kaggle 负责主要训练环境，本地只用于阅读代码、运行测试和 CPU 小规模验证。

## Kaggle 快速开始

先安装 Kaggle CLI 并完成登录：

```bash
uv tool install kaggle
kaggle auth login
```

然后阅读 [Kaggle 训练指南](docs/guides/kaggle.zh-CN.md)，修改其中的 GitHub 仓库地址和 Kaggle 用户名，提交：

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <你的用户名>/pytorch-text-classification-lab-ag-news-gpu
```

训练结束后下载产物：

```bash
kaggle kernels output <你的用户名>/pytorch-text-classification-lab-ag-news-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

Kaggle runner 会自动完成数据下载、manifest 生成、dry run、GPU 训练和测试集评估。Kaggle 的临时磁盘会在会话结束后清理，因此必须下载 `artifacts/`。

## 本地最短检查

```bash
uv sync --locked --extra dev
uv run text-classify show-config --config configs/learning_minimal.yaml
uv run text-classify train --config configs/learning_minimal.yaml --dry-run
uv run pytest
```

本地完整数据训练不是主线。需要本地准备数据时：

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run text-classify prepare-data --data-dir data/raw --manifest-dir data/manifests
```

## 学习路线

- `examples/01_tokens.py`：文本、token 和词表。
- `examples/02_padding_and_mask.py`：变长序列、padding 和 mask。
- `examples/03_minimal_training_loop.py`：logits、loss、反向传播和参数更新。
- `configs/learning_minimal.yaml`：CPU 可运行的最小配置。
- `configs/reference_textcnn.yaml`：Kaggle T4 参考训练配置。
- `configs/reference_bilstm.yaml`：序列模型对照配置。

第一版模型包括 `embedding_bag`、`text_cnn` 和 `bilstm`。Transformer 和预训练模型放在后续阶段，先保证基础数据流和评估协议清晰。

## 目录

```text
configs/                  配置文件
scripts/                  数据下载和检查脚本
docs/                     教程、Kaggle 流程和参考运行
examples/                 可单独运行的学习示例
src/text_classifier/      可安装的应用代码
tests/                    单元、集成和 CLI 测试
```

原始数据不提交到仓库。发布结果时请同时记录数据来源、许可证、manifest hash、词表 hash、配置、依赖和 Kaggle GPU 环境。

## 开发

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv build
uv run twine check dist/*
```

项目采用 MIT License。
