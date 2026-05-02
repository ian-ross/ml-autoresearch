import pytest
import torch

from ml_autoresearch.metrics import binary_segmentation_metrics


def test_binary_segmentation_metrics_on_known_masks():
    predicted = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float32)

    metrics = binary_segmentation_metrics(predicted, target, epsilon=0.0)

    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_binary_segmentation_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        binary_segmentation_metrics(torch.zeros((1, 1, 2, 2)), torch.zeros((1, 1, 2, 3)))
