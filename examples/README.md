# Small Examples

[中文](README.zh-CN.md) | [Tutorial](../docs/tutorial/README.md)

Each example demonstrates one concept. The first four use CPU; the last requires a trusted checkpoint.

| File | Concept | Prerequisite |
| --- | --- | --- |
| `01_tokens.py` | Unicode tokens, vocabulary, special IDs | None |
| `02_padding_and_mask.py` | Dynamic padding and attention mask | None |
| `03_minimal_training_loop.py` | Logits, cross-entropy, backward | None |
| `04_prepare_data.py` | Configuration and manifest APIs | Downloaded AG News |
| `05_checkpoint_prediction.py` | Real checkpoint loading and top-k | Trusted `best.pt` |

```bash
uv run python examples/01_tokens.py
uv run python examples/02_padding_and_mask.py
uv run python examples/03_minimal_training_loop.py
uv run python examples/04_prepare_data.py
uv run python examples/05_checkpoint_prediction.py \
  --checkpoint artifacts/tutorial-run/best.pt --text "The team won the final."
```

Examples are not separate implementations; they call the same package APIs as the CLI. Use `text-classify train` for complete training.
