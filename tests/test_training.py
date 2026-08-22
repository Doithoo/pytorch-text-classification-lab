import pytest
import torch
from torch import nn

from text_classifier.config import load_config
from text_classifier.training.train import _build_optimizer, resolve_device


@pytest.mark.parametrize(
    ("name", "optimizer_type"),
    [
        ("adamw", torch.optim.AdamW),
        ("adam", torch.optim.Adam),
        ("sgd", torch.optim.SGD),
    ],
)
def test_configured_optimizer_is_constructed(name: str, optimizer_type: type[torch.optim.Optimizer]) -> None:
    config = load_config(None, [f"train.optimizer={name}"])
    optimizer = _build_optimizer(nn.Linear(2, 2), config)
    assert isinstance(optimizer, optimizer_type)


def test_unavailable_cuda_fails_before_training() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
