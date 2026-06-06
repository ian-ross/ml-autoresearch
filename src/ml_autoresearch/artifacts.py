"""Harness-owned qualitative prediction sample artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.problem_support.frame_sequences import infer_timestamped_frame_sequences
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
) -> dict[str, object]:
    """Write bounded qualitative Contrail Mask prediction samples for a Run."""

    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if prediction_sample_policy not in {"first_n", "adjacent_and_scattered"}:
        raise ValueError(f"unsupported prediction sample policy: {prediction_sample_policy}")

    root = Path(run_dir)
    samples_dir = root / "outputs" / "prediction_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dataset_samples = getattr(data_loader.dataset, "samples", None)
    if isinstance(dataset_samples, list):
        selections = select_prediction_sample_indices(dataset_samples, policy=prediction_sample_policy, max_samples=max_samples)
    else:
        selections = select_prediction_sample_indices(
            list(range(len(data_loader.dataset))), policy="first_n", max_samples=max_samples
        )
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
            image = _display_rgb_from_model_input(inputs.detach().cpu().clamp(0.0, 1.0))
            target = target.detach().cpu() >= 0.5

            save_rgb_tensor(samples_dir / paths["input"], image)
            save_mask_tensor(samples_dir / paths["ground_truth"], target)
            save_mask_tensor(samples_dir / paths["prediction"], prediction)
            save_overlay(samples_dir / paths["overlay"], image, target, prediction)
            save_probability_heatmap(samples_dir / paths["probability_heatmap"], probabilities)

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
    if policy != "adjacent_and_scattered":
        raise ValueError(f"unsupported prediction sample policy: {policy}")
    return _select_adjacent_and_scattered(samples, max_samples=max_samples)


def _select_adjacent_and_scattered(samples: list[object], *, max_samples: int) -> list[dict[str, Any]]:
    index_by_identity = {id(sample): index for index, sample in enumerate(samples)}
    sequences = infer_timestamped_frame_sequences(samples, filename_for_item=lambda sample: getattr(sample, "image_path", ""))
    eligible_sequences = [sequence for sequence in sequences if len(sequence) >= 2 and any(_is_positive(sample) for sample in sequence)]
    selections: list[dict[str, Any]] = []

    scattered_budget = 0 if max_samples < 3 else max(1, min(2, max_samples // 3))
    adjacent_budget = max_samples - scattered_budget
    if eligible_sequences and adjacent_budget >= 2:
        window_length = 3 if adjacent_budget >= 6 and any(len(sequence) >= 3 for sequence in eligible_sequences) else 2
        window_count = max(1, adjacent_budget // window_length)
        for window_number, sequence_position in enumerate(_spread_positions(len(eligible_sequences), window_count)):
            if len(selections) + 2 > adjacent_budget:
                break
            sequence = eligible_sequences[sequence_position]
            window = _positive_adjacent_window(sequence, window_length)
            frame_sequence_id = _frame_sequence_id(sequence)
            window_id = f"{frame_sequence_id}/window_{window_number:03d}"
            for offset, sample in enumerate(window):
                if len(selections) >= adjacent_budget:
                    break
                selections.append(
                    _selection(
                        index_by_identity[id(sample)],
                        "adjacent_window",
                        frame_sequence_id=frame_sequence_id,
                        adjacent_window_id=window_id,
                        window_offset=offset,
                    )
                )

    remaining_budget = max_samples - len(selections)
    if remaining_budget > 0:
        already_selected = {int(selection["dataset_index"]) for selection in selections}
        selections.extend(_scattered_singletons(samples, remaining_budget, already_selected=already_selected))
    return selections[:max_samples]


def _positive_adjacent_window(sequence: list[object], window_length: int) -> list[object]:
    window_length = min(window_length, len(sequence))
    for start in range(0, len(sequence) - window_length + 1):
        window = sequence[start : start + window_length]
        if any(_is_positive(sample) for sample in window):
            return window
    return sequence[:window_length]


def _scattered_singletons(
    samples: list[object], budget: int, *, already_selected: set[int]
) -> list[dict[str, Any]]:
    available_positive = [index for index, sample in enumerate(samples) if index not in already_selected and _is_positive(sample)]
    available_negative = [index for index, sample in enumerate(samples) if index not in already_selected and not _is_positive(sample)]
    negative_count = 1 if budget >= 2 and available_negative else 0
    positive_count = min(len(available_positive), budget - negative_count)
    indices = _spread_values(available_positive, positive_count)
    indices.extend(_spread_values(available_negative, min(negative_count, budget - len(indices))))
    if len(indices) < budget:
        remaining = [index for index in range(len(samples)) if index not in already_selected and index not in indices]
        indices.extend(_spread_values(remaining, budget - len(indices)))
    return [_selection(index, "scattered_singleton") for index in indices[:budget]]


def _spread_positions(count: int, requested: int) -> list[int]:
    if count <= 0 or requested <= 0:
        return []
    if requested >= count:
        return list(range(count))
    if requested == 1:
        return [0]
    return [round(position * (count - 1) / (requested - 1)) for position in range(requested)]


def _spread_values(values: list[int], requested: int) -> list[int]:
    return [values[position] for position in _spread_positions(len(values), requested)]


def _selection(dataset_index: int, selection_kind: str, **extra: object) -> dict[str, Any]:
    payload: dict[str, Any] = {"dataset_index": dataset_index, "selection_kind": selection_kind}
    payload.update(extra)
    return payload


def _frame_sequence_id(sequence: list[object]) -> str:
    return Path(getattr(sequence[0], "image_path")).stem


def _is_positive(sample: object) -> bool:
    return bool(getattr(sample, "segmentations", ()))


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _display_rgb_from_model_input(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 3 and inputs.shape[0] == 3:
        return inputs
    if inputs.ndim == 3 and inputs.shape[0] == 9:
        return inputs[3:6]
    raise ValueError(f"cannot render prediction sample input with shape {tuple(inputs.shape)} as RGB")


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
