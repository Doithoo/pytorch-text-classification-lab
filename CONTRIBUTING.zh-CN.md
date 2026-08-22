# 贡献指南

[English](CONTRIBUTING.md)

## 环境

```bash
uv sync --locked --extra dev
uv run pre-commit install
```

请在独立分支提交范围明确的修改。行为变化必须带测试；公开命令、配置或文件协议变化必须同步更新中英文文档。不要提交原始数据、checkpoint、Kaggle 凭据、私有文本或与修改无关的生成文件。

## 验证

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
uv run pre-commit run --all-files
uv build
uv run twine check dist/*
```

测试不应下载 AG News、依赖 GPU、读取用户主目录或加载外部 checkpoint。文档命令应从仓库根目录执行。目录职责见 `src/README.zh-CN.md`、`scripts/README.zh-CN.md` 和 `tests/README.zh-CN.md`。

## 数据与实验

数据或实验修改需记录源 URL 与校验和、manifest identity、标签顺序、完整配置、依赖锁、精确 git revision 和命令。只用验证集选择模型，最终测试集只评估一次。数据许可与隐私责任不能由代码的 MIT License 替代。

## 提交与 PR

提交信息使用英文 ASCII 和 [Conventional Commits](https://www.conventionalcommits.org/) 格式，例如 `fix(models): Handle short TextCNN inputs`。主题保持简洁，不以句号结尾。

PR 说明应包含问题、行为变化、测试证据、兼容性影响和文档变化。checkpoint、manifest、配置或 CLI 的不兼容修改必须给迁移方案。
