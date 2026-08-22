import torch

from text_classifier.models import BiLSTMClassifier, EmbeddingBagClassifier, TextCNNClassifier


def test_models_return_class_logits() -> None:
    ids = torch.randint(0, 20, (3, 8))
    mask = torch.ones_like(ids)
    for model in [
        EmbeddingBagClassifier(20, 4, 8),
        TextCNNClassifier(20, 4, 8, kernel_sizes=[3, 4]),
        BiLSTMClassifier(20, 4, 8, 6, num_layers=1),
    ]:
        assert model(ids, mask).shape == (3, 4)
