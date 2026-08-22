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


def test_textcnn_accepts_sequences_shorter_than_its_kernels() -> None:
    model = TextCNNClassifier(20, 4, 8, kernel_sizes=[3, 4, 5])
    model.eval()
    for length in (2, 3, 4):
        ids = torch.randint(1, 20, (2, length))
        assert model(ids, torch.ones_like(ids)).shape == (2, 4)


def test_textcnn_ignores_trailing_padding_content() -> None:
    torch.manual_seed(7)
    model = TextCNNClassifier(20, 4, 8, kernel_sizes=[2, 3])
    model.eval()
    short_ids = torch.tensor([[2, 7, 8, 3]])
    short_mask = torch.ones_like(short_ids)
    padded_ids = torch.tensor([[2, 7, 8, 3, 12, 15]])
    padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    assert torch.allclose(model(short_ids, short_mask), model(padded_ids, padded_mask))
