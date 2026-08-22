# 脚本

[English](README.md) | [文档首页](../docs/README.zh-CN.md)

| 脚本 | 作用 | 是否联网 |
| --- | --- | --- |
| `download_data.py` | 下载固定 AG News CSV 并校验 SHA-256 | 是 |
| `generate_doc_assets.py` | 从参考运行生成训练曲线和混淆矩阵 | 否 |

下载数据：

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

若目标文件已存在，脚本先校验哈希；内容不匹配时失败。`--force` 重新下载，但仍不会跳过哈希。

重新生成文档图片需要可选绘图依赖：

```bash
uv run --extra plot python scripts/generate_doc_assets.py \
  --run-dir docs/recorded-run/kaggle-agnews-textcnn --output-dir docs/assets
```

图片必须来自提交的机器可读指标，不手工修改图中的数字。
