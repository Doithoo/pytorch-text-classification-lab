# AG News Three-Model Comparison Reference Run

[中文](README.zh-CN.md) | [Recorded-run index](../README.md)

This is the first complete three-model comparison from the current `0.3.0` code. It was submitted by `docs/recorded-run/kaggle-comparison/run_kaggle_comparison.py` at exact revision:

```text
0b350f27f3af6d1cbb417f064f3c546a9af11b46
```

All three runs use one AG News manifest, one tokenizer, one seed, eight epochs, and a Tesla T4. Model selection uses validation macro-F1 only, then each `best.pt` is evaluated on the same test split.

| Model | Best epoch | Best valid macro-F1 | Test accuracy | Test macro-F1 | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| BiLSTM | 4 | **0.924121** | **0.916053** | **0.915985** | 229.9 s |
| EmbeddingBag | 6 | 0.920197 | 0.915395 | 0.915278 | 61.4 s |
| TextCNN | 5 | 0.915575 | 0.910132 | 0.910090 | 159.0 s |

Total Kaggle runner time was 514.6 seconds on Tesla T4, PyTorch `2.10.0+cu128`, CUDA `12.8`, Python `3.12.13`.

## Data and protocol

```text
train / valid / test = 108,000 / 12,000 / 7,600
manifest_identity = 46780f09619b8d57e203a2719b2eec3ec3ea2b2a4039619a0673a1488ebb3447
tokenizer_sha256 = 1797a6c4c25c25ba9dfdf881e32b3baf63ac59d72ebca5f1d160d9169985bb39
```

Source SHA-256 matches the earlier AG News reference run:

```text
train.csv  76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e
test.csv   521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699
```

Test error counts are 638 for BiLSTM, 643 for EmbeddingBag, and 683 for TextCNN; counts and aggregate per-class metrics are retained, but per-example error text is not committed. Configs, epoch metrics, and run metadata remain; tokenizer and checkpoints are available from Kaggle output.

## Interpretation boundary

BiLSTM has the highest test macro-F1 under the same data and vocabulary, but takes about 3.7 times the EmbeddingBag training time. EmbeddingBag nearly matches it, showing that lexical/topic signals make a strong baseline for AG News. The current TextCNN score is below the older revision's `0.914610`; this cannot be attributed to one code change alone. The new masking behavior and independent config make this record the current comparison, while the older record remains historical evidence.

This is not a general benchmark across datasets or hardware. Model selection, balanced classes, AG News label boundaries, and one random seed all affect the result.

Kaggle kernel: https://www.kaggle.com/code/yashowhoo/pytorch-text-classification-lab-model-comparison
