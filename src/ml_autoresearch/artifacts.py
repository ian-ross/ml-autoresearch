"""Harness-owned qualitative prediction sample artifact generation."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.problem_support.imaging import (
    save_mask_tensor,
    save_overlay,
    save_probability_heatmap,
    save_rgb_tensor,
)
from ml_autoresearch.smoke import _extract_mask_logits


def write_prediction_sample_artifacts(
    *,
    run_dir: str | Path,
    model: torch.nn.Module,
    data_loader: DataLoader,
    split: str,
    max_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    output_spec: dict[str, object] | None = None,
    sample_selector: Callable[[object], list[dict[str, Any]]] | None = None,
    display_input_renderer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    sample_artifact_writer: Callable[[Path, dict[str, str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]
    | None = None,
) -> dict[str, object]:
    """Write bounded qualitative prediction samples for a Run."""

    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if prediction_sample_policy != "first_n" and sample_selector is None:
        raise ValueError(f"unsupported prediction sample policy: {prediction_sample_policy}")

    root = Path(run_dir)
    samples_dir = root / "outputs" / "prediction_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    if sample_selector is not None:
        selections = sample_selector(data_loader.dataset)
    else:
        dataset_samples = getattr(data_loader.dataset, "samples", None)
        selections = select_prediction_sample_indices(
            dataset_samples if isinstance(dataset_samples, list) else list(range(len(data_loader.dataset))),
            policy=prediction_sample_policy,
            max_samples=max_samples,
        )
    selections = selections[:max_samples]
    sample_records: list[dict[str, Any]] = []
    model.eval()
    device = _model_device(model)
    with torch.no_grad():
        for seen, selection in enumerate(selections):
            dataset_index = int(selection["dataset_index"])
            inputs, target = data_loader.dataset[dataset_index]
            model_inputs = inputs.unsqueeze(0).to(device)
            logits = _extract_mask_logits(model(model_inputs), output_spec)[0]
            probabilities = torch.sigmoid(logits).detach().cpu()[0]
            prediction = probabilities >= 0.5

            prefix = f"sample_{seen:03d}"
            paths = {
                "input": f"{prefix}_input.png",
                "ground_truth": f"{prefix}_ground_truth.png",
                "prediction": f"{prefix}_prediction.png",
                "overlay": f"{prefix}_overlay.png",
                "probability_heatmap": f"{prefix}_probability_heatmap.png",
            }
            renderer = display_input_renderer or _display_rgb_from_model_input
            image = renderer(inputs.detach().cpu().clamp(0.0, 1.0))
            target = target.detach().cpu() >= 0.5

            writer = sample_artifact_writer or _write_binary_segmentation_sample_images
            writer(samples_dir, paths, image, target, prediction, probabilities)

            metrics = binary_segmentation_metrics(prediction.unsqueeze(0), target.unsqueeze(0))
            record: dict[str, Any] = {
                "sample_id": f"{split}/{seen:06d}",
                "split": split,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "paths": paths,
                "selection": selection,
            }
            source_image_path = _source_image_path(data_loader.dataset, dataset_index)
            if source_image_path is not None:
                record["source_image_path"] = source_image_path
            sample_records.append(record)

    manifest = {
        "status": "completed",
        "split": split,
        "prediction_sample_policy": prediction_sample_policy,
        "sample_count": len(sample_records),
        "max_sample_count": max_samples,
        "samples": sample_records,
    }
    (samples_dir / "samples.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"prediction_samples": "outputs/prediction_samples/samples.json"}


def select_prediction_sample_indices(samples: object, *, policy: str, max_samples: int) -> list[dict[str, Any]]:
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if not isinstance(samples, list):
        return [_selection(index, "first_n") for index in range(max_samples)]
    if policy == "first_n":
        return [_selection(index, "first_n") for index in range(min(max_samples, len(samples)))]
    raise ValueError(f"unsupported prediction sample policy: {policy}")


def _selection(dataset_index: int, selection_kind: str, **extra: object) -> dict[str, Any]:
    payload: dict[str, Any] = {"dataset_index": dataset_index, "selection_kind": selection_kind}
    payload.update(extra)
    return payload


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _display_rgb_from_model_input(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 3 and inputs.shape[0] == 3:
        return inputs
    raise ValueError(f"cannot render prediction sample input with shape {tuple(inputs.shape)} as RGB")


def _write_binary_segmentation_sample_images(
    samples_dir: Path,
    paths: dict[str, str],
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    save_rgb_tensor(samples_dir / paths["input"], image)
    save_mask_tensor(samples_dir / paths["ground_truth"], target)
    save_mask_tensor(samples_dir / paths["prediction"], prediction)
    save_overlay(samples_dir / paths["overlay"], image, target, prediction)
    save_probability_heatmap(samples_dir / paths["probability_heatmap"], probabilities)


def _source_image_path(dataset: object, index: int) -> str | None:
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list) and index < len(samples):
        image_path = getattr(samples[index], "image_path", None)
        if image_path is not None:
            return str(Path(image_path))
    return None


_save_rgb_tensor = save_rgb_tensor
_save_mask_tensor = save_mask_tensor
_save_overlay = save_overlay
_save_probability_heatmap = save_probability_heatmap
