# Kaggle Training Guide

[中文](kaggle.zh-CN.md) | [Documentation index](../README.md)

The full training environment is one Kaggle Tesla T4. The runner uses `cuda:0` and does not depend on multiple GPUs.

## Prepare

Install and authenticate the Kaggle CLI:

```bash
uv tool install kaggle
kaggle auth login
```

Edit `docs/recorded-run/kaggle/kernel-metadata.json` and replace `yashowhoo` in `id` with your Kaggle account. Keep GPU, Internet, and T4 settings. The runner clones a public repository and downloads AG News, so the default branch must be accessible. For a fork, also update `PROJECT_URL` in `run_kaggle.py`.

## Submit and observe

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <your-user>/pytorch-text-classification-lab-ag-news-gpu
```

The sequence is:

```text
git clone -> install -> download -> prepare -> inspect -> dry run -> train -> evaluate
```

The log first asserts CUDA, then performs a minimal dry run, then trains `reference_textcnn.yaml`. The recorded run took about 2.5 minutes, but queues, network, and Kaggle images change wall time.

## Download evidence

After status becomes `COMPLETE`:

```bash
kaggle kernels output <your-user>/pytorch-text-classification-lab-ag-news-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

Retain `best.pt`, `last.pt`, `config.yaml`, `tokenizer.json`, `metrics.csv`, `run.json`, evaluation, and `kaggle-run-summary.json`. `/kaggle/working` is temporary.

## Interruptions and resume

Restore the previous complete run directory and increase total epochs:

```bash
text-classify train --config configs/reference_textcnn.yaml \
  --resume artifacts/kaggle-agnews-textcnn/last.pt \
  --set train.epochs=12 --set device=cuda
```

Never load an untrusted checkpoint. Resume validates manifest, model, tokenizer, and important optimizer settings.

## Three-model comparison

`docs/recorded-run/kaggle-comparison/` provides a current-code runner for EmbeddingBag, TextCNN, and BiLSTM on one manifest:

```bash
kaggle kernels push -p docs/recorded-run/kaggle-comparison
```

It writes three run directories and identity-validated `comparison.json`. The `0.3.0` run is complete; see the [published evidence](../recorded-run/kaggle-agnews-model-comparison-v0.3.0/README.md). Future submissions should use a new versioned directory and never replace an existing result.

## Publish a result

Select settings on validation, then evaluate test once. A publication page should include exact git revision, dataset and manifest identity, resolved config, dependency environment, GPU, wall time, full metrics, and errors. Do not publish estimated metrics.
