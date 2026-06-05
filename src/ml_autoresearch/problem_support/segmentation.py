"""Reusable trusted binary segmentation support functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

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


def binary_confusion_counts(predicted_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, int]:
    """Return pixel confusion counts for binary segmentation masks."""

    pred = predicted_mask.bool()
    target = target_mask.bool()
    if pred.shape != target.shape:
        raise ValueError(f"predicted_mask and target_mask shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}")
    return {
        "positive_pixel_count": int(target.sum().item()),
        "predicted_positive_pixel_count": int(pred.sum().item()),
        "true_positive_pixels": int(torch.logical_and(pred, target).sum().item()),
        "false_positive_pixels": int(torch.logical_and(pred, ~target).sum().item()),
        "false_negative_pixels": int(torch.logical_and(~pred, target).sum().item()),
    }


def bce_dice_loss(mask_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy plus soft Dice loss for mask logits."""

    bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
    probabilities = torch.sigmoid(mask_logits)
    intersection = (probabilities * target_mask).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + _EPSILON) / (denominator + _EPSILON)).mean()
    return bce + dice_loss


def derive_line_target_v1(target_mask: torch.Tensor) -> torch.Tensor:
    """Derive the v1 Line Target as a small tolerance band around positives."""

    return F.max_pool2d(target_mask.float(), kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)


def derive_boundary_target_v1(target_mask: torch.Tensor) -> torch.Tensor:
    """Derive the v1 Boundary Target as a deterministic one-pixel edge band."""

    mask = target_mask.float().clamp(0.0, 1.0)
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, *, positive_weight: float = 4.0) -> torch.Tensor:
    """Weighted BCE helper for auxiliary binary targets."""

    pos_weight = torch.tensor(float(positive_weight), dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
