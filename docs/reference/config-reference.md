# Configuration Reference

[中文](config-reference.zh-CN.md) | [Documentation index](../README.md)

Final configuration is defaults merged with YAML and `--set`, then validated before data access. Use `show-config` to inspect effective values.

## data

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `name` | `ag_news` | `ag_news` or headered `generic_csv` |
| `data_dir` | `data/raw` | AG News root or directory containing generic train/test CSV |
| `manifest_dir` | `data/manifests` | Prepared CSV, `dataset.json`, and audit directory |
| `tokenizer` | `simple_word` | Built-in word tokenizer only |
| `text_column` | `text` | Generic CSV text column |
| `label_column` | `label` | Generic CSV label column; must differ from text |
| `vocab_size` | `30000` | Includes four special tokens; at least 4 |
| `min_frequency` | `2` | Positive training-vocabulary frequency |
| `max_length` | `128` | Includes BOS/EOS; at least 2 |
| `valid_ratio` | `0.1` | Strictly between zero and one, per class |
| `num_workers` | `0` | Non-negative DataLoader workers |
| `max_*_samples` | `null` | Per-split debug limit; null or positive integer |

## model

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `name` | `embedding_bag` | `embedding_bag`, `text_cnn`, or `bilstm` |
| `embedding_dim` | `128` | Positive token-vector width |
| `hidden_dim` | `128` | CNN channels or LSTM units per direction |
| `dropout` | `0.2` | `[0,1)` |
| `kernel_sizes` | `[3,4,5]` | Positive TextCNN list, no larger than max length |
| `num_layers` | `2` | Positive BiLSTM layer count |
| `bidirectional` | `true` | Whether BiLSTM is bidirectional |

## train and run

| Field | Default | Constraint and meaning |
| --- | --- | --- |
| `epochs` | `2` | Total epochs; target total when resuming |
| `batch_size` | `32` | Positive integer |
| `lr` | `0.001` | Positive number |
| `weight_decay` | `0.0001` | Non-negative number |
| `optimizer` | `adamw` | `adamw`, `adam`, or `sgd` |
| `momentum` | `0.9` | SGD momentum; ignored by other optimizers |
| `seed` | `42` | Non-negative integer |
| `amp` | `false` | Automatic mixed precision on CUDA only |
| `deterministic` | `false` | Request deterministic algorithms |
| `grad_clip` | `0.0` | Clip gradient norm when greater than zero |
| `best_metric` | `macro_f1` | Accuracy or one of four macro metrics |
| `device` | `auto` | auto, cpu, mps, cuda, or cuda:N |
| `output_dir` | `artifacts` | Run root |
| `run_name` | `null` | Single path component; null creates timestamp name |

```bash
uv run text-classify show-config --config configs/reference_textcnn.yaml \
  --set train.epochs=12 --set train.optimizer=adamw
```
