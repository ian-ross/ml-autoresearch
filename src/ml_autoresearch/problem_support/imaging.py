"""Reusable trusted imaging helpers for segmentation Research Problems."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def rasterize_coco_polygons(*, width: int, height: int, segmentations: tuple[tuple[float, ...], ...]) -> Image.Image:
    """Rasterize COCO-style polygon segmentations into a binary mask image."""

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentations:
        points = list(zip(polygon[0::2], polygon[1::2], strict=True))
        draw.polygon(points, fill=255)
    return mask


def rgb_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a float CHW tensor in [0, 1]."""

    data = torch.from_numpy(np.array(image, copy=True))
    return data.permute(2, 0, 1).float().div(255.0)


def mask_image_to_tensor(mask: Image.Image) -> torch.Tensor:
    """Convert a mask PIL image to a binary 1HW float tensor."""

    data = torch.from_numpy(np.array(mask, copy=True))
    return (data > 0).unsqueeze(0).float()


def horizontal_flip_image_mask(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip CHW image and 1HW mask tensors horizontally."""

    return torch.flip(image, dims=[2]), torch.flip(mask, dims=[2])


def deterministic_photometric_perturbation(
    image: torch.Tensor,
    *,
    contrast: float,
    brightness: float,
    noise_seed: int,
    noise_scale: float = 0.005,
) -> torch.Tensor:
    """Apply deterministic brightness/contrast and sensor-noise perturbation."""

    adjusted = (image - 0.5) * contrast + 0.5 + brightness
    generator = torch.Generator(device=image.device).manual_seed(int(noise_seed))
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype) * float(noise_scale)
    return torch.clamp(adjusted + noise, 0.0, 1.0)


def save_rgb_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    """Save a CHW RGB tensor in [0, 1] as an image."""

    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def save_mask_tensor(path: str | Path, tensor: torch.Tensor) -> None:
    """Save a binary mask tensor as a single-channel image."""

    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    array = tensor.numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


def save_overlay(path: str | Path, image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor) -> None:
    """Save an RGB overlay of target and prediction masks."""

    base = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    overlay = base.astype(np.float32)
    target_mask = target.squeeze(0).numpy().astype(bool)
    prediction_mask = prediction.squeeze(0).numpy().astype(bool)
    overlay[target_mask] = 0.55 * overlay[target_mask] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
    overlay[prediction_mask] = 0.55 * overlay[prediction_mask] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def save_probability_heatmap(path: str | Path, probabilities: torch.Tensor) -> None:
    """Save scalar mask probabilities as an RGB heatmap."""

    if probabilities.ndim == 3:
        probabilities = probabilities.squeeze(0)
    values = probabilities.detach().cpu().clamp(0.0, 1.0).numpy().astype(np.float32)
    heatmap = np.stack(
        [
            values,
            1.0 - np.abs((values * 2.0) - 1.0),
            1.0 - values,
        ],
        axis=-1,
    )
    Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path)
