"""Harness-owned qualitative prediction sample artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.smoke import _extract_mask_logits


def write_prediction_sample_artifacts(
    *,
    run_dir: str | Path,
    model: torch.nn.Module,
    data_loader: DataLoader,
    split: str,
    max_samples: int = 2,
) -> dict[str, object]:
    """Write bounded qualitative Contrail Mask prediction samples for a Run."""

    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")

    root = Path(run_dir)
    samples_dir = root / "outputs" / "prediction_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample_records: list[dict[str, Any]] = []
    model.eval()
    device = _model_device(model)
    seen = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            model_inputs = inputs.to(device)
            logits = _extract_mask_logits(model(model_inputs))[0]
            probabilities = torch.sigmoid(logits).detach().cpu()
            predictions = probabilities >= 0.5
            for item_index in range(inputs.shape[0]):
                if seen >= max_samples:
                    break
                prefix = f"sample_{seen:03d}"
                paths = {
                    "input": f"{prefix}_input.png",
                    "ground_truth": f"{prefix}_ground_truth.png",
                    "prediction": f"{prefix}_prediction.png",
                    "overlay": f"{prefix}_overlay.png",
                }
                image = inputs[item_index].detach().cpu().clamp(0.0, 1.0)
                target = targets[item_index].detach().cpu() >= 0.5
                prediction = predictions[item_index].detach().cpu()

                _save_rgb_tensor(samples_dir / paths["input"], image)
                _save_mask_tensor(samples_dir / paths["ground_truth"], target)
                _save_mask_tensor(samples_dir / paths["prediction"], prediction)
                _save_overlay(samples_dir / paths["overlay"], image, target, prediction)

                metrics = binary_segmentation_metrics(prediction.unsqueeze(0), target.unsqueeze(0))
                record: dict[str, Any] = {
                    "sample_id": f"{split}/{seen:06d}",
                    "split": split,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "paths": paths,
                }
                source_image_path = _source_image_path(data_loader.dataset, seen)
                if source_image_path is not None:
                    record["source_image_path"] = source_image_path
                sample_records.append(record)
                seen += 1
            if seen >= max_samples:
                break

    manifest = {
        "status": "completed",
        "split": split,
        "sample_count": len(sample_records),
        "max_sample_count": max_samples,
        "samples": sample_records,
    }
    (samples_dir / "samples.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"prediction_samples": "outputs/prediction_samples/samples.json"}


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _source_image_path(dataset: object, index: int) -> str | None:
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list) and index < len(samples):
        image_path = getattr(samples[index], "image_path", None)
        if image_path is not None:
            return str(Path(image_path))
    return None


def _save_rgb_tensor(path: Path, tensor: torch.Tensor) -> None:
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def _save_mask_tensor(path: Path, tensor: torch.Tensor) -> None:
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    array = tensor.numpy().astype(np.uint8) * 255
    Image.fromarray(array).save(path)


def _save_overlay(path: Path, image: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor) -> None:
    base = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    overlay = base.astype(np.float32)
    target_mask = target.squeeze(0).numpy().astype(bool)
    prediction_mask = prediction.squeeze(0).numpy().astype(bool)
    overlay[target_mask] = 0.55 * overlay[target_mask] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
    overlay[prediction_mask] = 0.55 * overlay[prediction_mask] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)
