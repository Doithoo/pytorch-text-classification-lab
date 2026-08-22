import torch

from text_classifier.data.dataset import collate_texts
from text_classifier.data.tokenizer import SimpleWordTokenizer


def test_tokenizer_builds_only_known_words() -> None:
    tokenizer = SimpleWordTokenizer.build(["Hello world", "hello"], vocab_size=16, min_frequency=1, max_length=8)
    assert tokenizer.encode("hello unknown")[0] == tokenizer.vocab["<bos>"]
    assert tokenizer.encode("hello unknown")[-1] == tokenizer.vocab["<eos>"]
    assert tokenizer.unk_id in tokenizer.encode("hello unknown")


def test_collate_pads_and_masks() -> None:
    batch = collate_texts(
        [
            {"id": "a", "text": "a", "input_ids": [2, 4, 3], "label": 0},
            {"id": "b", "text": "b", "input_ids": [2, 5, 6, 3], "label": 1},
        ]
    )
    assert batch["input_ids"].shape == (2, 4)
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]))
