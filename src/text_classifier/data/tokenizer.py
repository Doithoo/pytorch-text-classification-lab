from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


class SimpleWordTokenizer:
    def __init__(self, vocab: dict[str, int], max_length: int = 128) -> None:
        self.vocab = vocab
        self.max_length = max_length
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.casefold())

    @classmethod
    def build(cls, texts: list[str], vocab_size: int, min_frequency: int, max_length: int) -> SimpleWordTokenizer:
        counts = Counter(token for text in texts for token in cls.tokenize(text))
        words = sorted(
            (word for word, count in counts.items() if count >= min_frequency), key=lambda w: (-counts[w], w)
        )
        words = words[: max(0, vocab_size - len(SPECIAL_TOKENS))]
        vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        vocab.update({token: index for index, token in enumerate(words, start=len(SPECIAL_TOKENS))})
        return cls(vocab, max_length)

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)[: max(0, self.max_length - 2)]
        ids = [self.vocab["<bos>"]]
        ids.extend(self.vocab.get(token, self.unk_id) for token in tokens)
        ids.append(self.vocab["<eos>"])
        return ids

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"vocab": self.vocab, "max_length": self.max_length}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SimpleWordTokenizer:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in raw["vocab"].items()}, int(raw["max_length"]))

    def metadata(self) -> dict[str, object]:
        return {
            "name": "simple_word",
            "vocab_size": len(self.vocab),
            "max_length": self.max_length,
            "vocab": self.vocab,
        }
