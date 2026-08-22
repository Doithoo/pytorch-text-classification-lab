# AG News TextCNN Kaggle Reference Run

[中文](README.zh-CN.md) | [Recorded-run index](../README.md)

This first real Kaggle GPU run was submitted by `docs/recorded-run/kaggle/run_kaggle.py` at exact revision:

```text
4ed733b532b9169816d02797fece33e34135bc20
```

| Item | Result |
| --- | ---: |
| Dataset | AG News |
| Train / valid / test | 108,000 / 12,000 / 7,600 |
| Model | TextCNN, embedding 256, channels 128, kernels 3/4/5 |
| Device | Kaggle Tesla T4 |
| PyTorch / CUDA / Python | 2.10.0+cu128 / 12.8 / 3.12.13 |
| Epochs | 8 |
| Best epoch | 3 |
| Best validation macro-F1 | **0.915909** |
| Test accuracy | **0.914605** |
| Test macro-F1 | **0.914610** |
| Kaggle wall clock | 151.3 seconds |

![Training and validation curves](../../assets/ag-news-textcnn-training.png)

Training loss continued down after the best epoch while validation declined, showing overfitting. Test therefore used `best.pt`, not `last.pt`.

![Test confusion matrix](../../assets/ag-news-textcnn-confusion.png)

Among 649 test errors, the largest two-way confusion was business -> sci_tech (155) and sci_tech -> business (129). 186 mistakes had confidence at least 0.9. High-confidence errors include sports-person stories and international events with ambiguous editorial boundaries, so not every error is simply model capacity.

Source SHA-256:

```text
train.csv  76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e
test.csv   521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699
```

Manifest identity:

```text
46780f09619b8d57e203a2719b2eec3ec3ea2b2a4039619a0673a1488ebb3447
```

The repository retains `config.yaml`, `metrics.csv`, `run.json`, aggregate test metrics, and Kaggle summary. It does not commit the tokenizer or per-example error text, avoiding duplicate distribution of derived vocabulary and news text. The 93 MB `best.pt` and `last.pt` remain downloadable from Kaggle. This result belongs strictly to the revision above; current code may produce different values and must not silently replace it.

Kaggle notebook: https://www.kaggle.com/code/yashowhoo/pytorch-text-classification-lab-ag-news-gpu
