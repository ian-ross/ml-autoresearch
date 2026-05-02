"""Segmentation metric computations for Harness-owned evaluation."""

from __future__ import annotations

import torch

_EPSILON = 1e-7


def binary_segmentation_metrics(
    predicted_mask: torch.Tensor, target_mask: torch.Tensor, *, epsilon: float = _EPSILON
) -> dict[str, float]:
    """Compute binary segmentation metrics from boolean/0-1 masks."""

    pred = predicted_mask.bool()
    target = target_mask.bool()
    if pred.shape != target.shape:
        raise ValueError(f"predicted_mask and target_mask shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}")

    true_positive = torch.logical_and(pred, target).sum().item()
    false_positive = torch.logical_and(pred, ~target).sum().item()
    false_negative = torch.logical_and(~pred, target).sum().item()

    dice = (2 * true_positive + epsilon) / (2 * true_positive + false_positive + false_negative + epsilon)
    iou = (true_positive + epsilon) / (true_positive + false_positive + false_negative + epsilon)
    precision = (true_positive + epsilon) / (true_positive + false_positive + epsilon)
    recall = (true_positive + epsilon) / (true_positive + false_negative + epsilon)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }
