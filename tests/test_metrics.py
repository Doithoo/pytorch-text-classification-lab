import pytest

from text_classifier.evaluation.metrics import classification_metrics


def test_classification_metrics_reports_confusion_and_macro_values() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion"] == [[1, 1], [0, 2]]
    assert metrics["per_class_support"] == [2, 2]


@pytest.mark.parametrize(
    ("labels", "predictions", "classes", "message"),
    [
        ([0], [], 2, "equal length"),
        ([0], [0], 0, "positive"),
        ([-1], [0], 2, "valid class IDs"),
        ([0], [2], 2, "valid class IDs"),
    ],
)
def test_classification_metrics_rejects_invalid_inputs(
    labels: list[int], predictions: list[int], classes: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        classification_metrics(labels, predictions, classes)
