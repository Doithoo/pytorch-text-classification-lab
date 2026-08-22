# AG News TextCNN Kaggle Reference Run

This is the first real Kaggle GPU run for the project. It was executed by the non-interactive runner at
`docs/recorded-run/kaggle/run_kaggle.py` using Git revision
`4ed733b532b9169816d02797fece33e34135bc20`.

| Item | Value |
| --- | ---: |
| Dataset | AG News |
| Train / valid / test | 108,000 / 12,000 / 7,600 |
| Model | TextCNN, embedding 256, channels 128, kernels 3/4/5 |
| Device | Kaggle Tesla T4 |
| PyTorch / CUDA | 2.10.0+cu128 / 12.8 |
| Python | 3.12.13 |
| Epochs | 8 |
| Best epoch | 3 |
| Best validation macro-F1 | **0.915909** |
| Test accuracy | **0.914605** |
| Test macro-F1 | **0.914610** |
| Wall clock | 151.3 seconds |

The run used the fixed source hashes in `data/` preparation:

```text
train.csv  76a0a2d2f92b286371fe4d4044640910a04a803fdd2538e0f3f29a5c6f6b672e
test.csv   521465c2428ed7f02f8d6db6ffdd4b5447c1c701962353eb2c40d548c3c85699
```

The prepared manifest identity recorded by the run is:

```text
46780f09619b8d57e203a2719b2eec3ec3ea2b2a4039619a0673a1488ebb3447
```

The repository keeps the configuration, tokenizer, training curve, test metrics, error samples, and Kaggle
summary. The 93 MB checkpoints remain downloadable from the Kaggle kernel output instead of being committed to
Git. The test confusion matrix and per-class metrics are in `evaluation/metrics.json`.

Kaggle notebook: https://www.kaggle.com/code/yashowhoo/pytorch-text-classification-lab-ag-news-gpu
