# Kaggle Training Guide

Kaggle GPU is the primary training environment for this repository. The runner targets a single NVIDIA T4 and uses `cuda:0`.

Before submitting, push the repository to GitHub and update `PROJECT_URL` in `docs/recorded-run/kaggle/run_kaggle.py`.
Install and authenticate the Kaggle CLI:

```bash
uv tool install kaggle
kaggle auth login
```

Replace the account name in `kernel-metadata.json`, then submit:

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <your-user>/pytorch-text-classification-lab-ag-news-gpu
```

The runner performs:

```text
git clone -> install -> download AG News -> prepare manifests -> inspect -> dry run -> CUDA training -> test evaluation
```

Download results before the temporary working volume disappears:

```bash
kaggle kernels output <your-user>/pytorch-text-classification-lab-ag-news-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

Keep the checkpoint together with its config, metrics, evaluation, errors, and `kaggle-run-summary.json`.
The project writes `last.pt` for resuming after a session interruption. Resume only after checking the tokenizer,
manifest identity, model config, and training parameters.
