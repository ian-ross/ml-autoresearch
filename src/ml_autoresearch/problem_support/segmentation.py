"""Reusable trusted binary segmentation support functions."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from ml_autoresearch.smoke import _extract_mask_logits, _import_candidate_model, input_spec_from_resolved_manifest, output_spec_from_resolved_manifest

_EPSILON = 1e-7
DEFAULT_BINARY_SEGMENTATION_THRESHOLD = 0.5


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


def binary_segmentation_validation_metrics(logits: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    """Return Harness validation metric names for binary segmentation logits."""

    metrics = binary_segmentation_metrics(torch.sigmoid(logits) >= 0.5, target_mask >= 0.5)
    return {
        "val/dice": metrics["dice"],
        "val/iou": metrics["iou"],
        "val/precision": metrics["precision"],
        "val/recall": metrics["recall"],
    }


def evaluate_binary_segmentation_validation_split(
    *,
    adapter: object,
    run_dir: Path,
    data_root: Path,
    model_artifact_path: Path,
    threshold: float,
    evaluation_dir: Path,
    max_artifact_samples: int,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Run Whole-Validation Failure Analysis for a binary segmentation Research Problem."""

    manifest = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    batch_size = int(manifest.get("training", {}).get("batch_size", 1))
    input_spec = input_spec_from_resolved_manifest(run_dir / "resolved_manifest.yaml")
    output_spec = output_spec_from_resolved_manifest(run_dir / "resolved_manifest.yaml")
    build_evaluation_dataset = getattr(adapter, "build_evaluation_dataset", None)
    if not callable(build_evaluation_dataset):
        raise RuntimeError("Research Problem does not provide an evaluation dataset adapter")
    dataset = build_evaluation_dataset(data_config={"dataset_root": str(data_root)}, resolved_manifest_path=run_dir / "resolved_manifest.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = _import_candidate_model(run_dir / "candidate")
    model = module.build_model(dict(input_spec), dict(output_spec))
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("build_model must return a torch.nn.Module")
    checkpoint = torch.load(model_artifact_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"model artifact is unreadable: missing model_state_dict in {model_artifact_path}")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    all_inputs: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    per_sample_records: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch_indices = list(range(start, min(start + batch_size, len(dataset))))
            inputs = torch.stack([dataset[index][0] for index in batch_indices]).to(device)
            targets = torch.stack([dataset[index][1] for index in batch_indices]).to(device)
            logits = _extract_mask_logits(model(inputs), output_spec)[0]
            probabilities = torch.sigmoid(logits).detach().cpu()
            predictions = probabilities >= threshold
            target_masks = (targets >= 0.5).detach().cpu()
            all_inputs.append(inputs.detach().cpu())
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
            all_targets.append(target_masks)
            for offset, index in enumerate(batch_indices):
                sample_prediction = predictions[offset : offset + 1]
                sample_target = target_masks[offset : offset + 1]
                metrics = binary_segmentation_metrics(sample_prediction, sample_target)
                sample = dataset.samples[index]
                image_path = getattr(sample, "image_path", "")
                per_sample_records.append(
                    {
                        "sample_id": f"val/{index:06d}",
                        "dataset_index": index,
                        "image_id": str(getattr(sample, "image_id", Path(str(image_path)).stem)),
                        "image_path": str(image_path),
                        **metrics,
                        **binary_confusion_counts(sample_prediction, sample_target),
                    }
                )
    input_tensor = torch.cat(all_inputs)
    prediction_tensor = torch.cat(all_predictions)
    probability_tensor = torch.cat(all_probabilities)
    target_tensor = torch.cat(all_targets)
    aggregate = binary_segmentation_metrics(prediction_tensor, target_tensor)
    threshold_sweep = summarize_binary_segmentation_threshold_sweep(
        probability_tensor,
        target_tensor,
        default_threshold=DEFAULT_BINARY_SEGMENTATION_THRESHOLD,
    )
    diagnostic_manifest = write_binary_segmentation_diagnostic_sample_artifacts(
        adapter=adapter,
        evaluation_dir=evaluation_dir,
        dataset=dataset,
        inputs=input_tensor,
        probabilities=probability_tensor,
        predictions=prediction_tensor,
        targets=target_tensor,
        per_sample_records=per_sample_records,
        threshold=threshold,
        max_artifact_samples=max_artifact_samples,
    )
    return aggregate, per_sample_records, threshold_sweep, diagnostic_manifest


def write_binary_segmentation_diagnostic_sample_artifacts(
    *,
    adapter: object,
    evaluation_dir: Path,
    dataset: object,
    inputs: torch.Tensor,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    per_sample_records: list[dict[str, object]],
    threshold: float,
    max_artifact_samples: int,
) -> dict[str, object]:
    """Write bounded diagnostic artifacts for binary segmentation evaluation records."""

    diagnostics_dir = evaluation_dir / "diagnostic_samples"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    selections = select_binary_segmentation_failure_bucket_indices(per_sample_records, max_artifact_samples=max_artifact_samples)
    samples: list[dict[str, object]] = []
    for artifact_index, selection in enumerate(selections):
        dataset_index = int(selection["dataset_index"])
        prefix = f"sample_{artifact_index:03d}"
        paths = {
            "input": f"{prefix}_input.png",
            "ground_truth": f"{prefix}_ground_truth.png",
            "prediction": f"{prefix}_prediction.png",
            "overlay": f"{prefix}_overlay.png",
            "probability_heatmap": f"{prefix}_probability_heatmap.png",
        }
        display_input = getattr(adapter, "display_prediction_sample_input", None)
        write_images = getattr(adapter, "write_prediction_sample_images", None)
        if not callable(display_input) or not callable(write_images):
            raise RuntimeError("Research Problem evaluation adapter cannot render diagnostic samples")
        image = display_input(inputs[dataset_index].detach().cpu().clamp(0.0, 1.0))
        target = targets[dataset_index].detach().cpu().bool()
        prediction = predictions[dataset_index].detach().cpu().bool()
        probability = probabilities[dataset_index].detach().cpu()
        write_images(diagnostics_dir, paths, image, target, prediction, probability)

        source = dataset.samples[dataset_index]
        samples.append(
            {
                "sample_id": f"val/{dataset_index:06d}",
                "dataset_index": dataset_index,
                "source_image_path": str(source.image_path),
                "threshold": threshold,
                "metrics": {
                    "dice": selection["dice"],
                    "iou": selection["iou"],
                    "precision": selection["precision"],
                    "recall": selection["recall"],
                },
                "positive_pixel_count": selection["positive_pixel_count"],
                "predicted_positive_pixel_count": selection["predicted_positive_pixel_count"],
                "true_positive_pixels": selection["true_positive_pixels"],
                "false_positive_pixels": selection["false_positive_pixels"],
                "false_negative_pixels": selection["false_negative_pixels"],
                "bucket_memberships": selection["bucket_memberships"],
                "paths": paths,
            }
        )
    return {
        "status": "completed",
        "split": "val",
        "threshold": threshold,
        "sample_count": len(samples),
        "max_artifact_samples": max_artifact_samples,
        "buckets": [
            "worst_by_dice",
            "best_by_dice",
            "false_positive_heavy",
            "false_negative_heavy",
            "empty_mask_false_positives",
            "missed_positive_masks",
        ],
        "samples": samples,
    }


def select_binary_segmentation_failure_bucket_indices(
    per_sample_records: list[dict[str, object]], *, max_artifact_samples: int
) -> list[dict[str, object]]:
    """Select bounded, deduplicated binary segmentation failure-bucket records."""

    if max_artifact_samples < 1:
        raise RuntimeError("max_artifact_samples must be at least 1")

    memberships: dict[int, set[str]] = {}

    def add(record: dict[str, object], bucket: str) -> None:
        index = int(record["dataset_index"])
        memberships.setdefault(index, set()).add(bucket)

    records = list(per_sample_records)
    positives = [record for record in records if int(record.get("positive_pixel_count", 0)) > 0]
    empty = [record for record in records if int(record.get("positive_pixel_count", 0)) == 0]
    predicted_positive = [record for record in records if int(record.get("predicted_positive_pixel_count", 0)) > 0]
    bucket_count = 6
    per_bucket_limit = max(1, max_artifact_samples // bucket_count)

    for record in sorted(records, key=lambda item: (float(item.get("dice", 0.0)), int(item["dataset_index"])))[:per_bucket_limit]:
        add(record, "worst_by_dice")
    for record in sorted(records, key=lambda item: (-float(item.get("dice", 0.0)), int(item["dataset_index"])))[:per_bucket_limit]:
        add(record, "best_by_dice")
    for record in sorted(
        [record for record in predicted_positive if int(record.get("false_positive_pixels", 0)) > 0],
        key=lambda item: (-int(item.get("false_positive_pixels", 0)), int(item["dataset_index"])),
    )[:per_bucket_limit]:
        add(record, "false_positive_heavy")
    for record in sorted(
        [record for record in positives if int(record.get("false_negative_pixels", 0)) > 0],
        key=lambda item: (-int(item.get("false_negative_pixels", 0)), int(item["dataset_index"])),
    )[:per_bucket_limit]:
        add(record, "false_negative_heavy")
    for record in sorted(
        [record for record in empty if int(record.get("false_positive_pixels", 0)) > 0],
        key=lambda item: (-int(item.get("false_positive_pixels", 0)), int(item["dataset_index"])),
    )[:per_bucket_limit]:
        add(record, "empty_mask_false_positives")
    for record in sorted(
        [record for record in positives if int(record.get("predicted_positive_pixel_count", 0)) == 0],
        key=lambda item: (-int(item.get("false_negative_pixels", 0)), int(item["dataset_index"])),
    )[:per_bucket_limit]:
        add(record, "missed_positive_masks")

    by_index = {int(record["dataset_index"]): record for record in records}
    ordered_indices = sorted(
        memberships,
        key=lambda index: (
            -len(memberships[index]),
            float(by_index[index].get("dice", 0.0)),
            index,
        ),
    )[:max_artifact_samples]
    selected: list[dict[str, object]] = []
    for index in ordered_indices:
        selected_record = dict(by_index[index])
        selected_record["bucket_memberships"] = sorted(memberships[index])
        selected.append(selected_record)
    return selected


def summarize_binary_segmentation_threshold_sweep(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    default_threshold: float,
) -> dict[str, object]:
    """Compute aggregate threshold diagnostics for binary segmentation probabilities."""

    target_masks = targets.bool()
    thresholds = [round(index * 0.05, 2) for index in range(1, 20)]
    groups = {
        "all_samples": torch.ones(target_masks.shape[0], dtype=torch.bool),
        "positive_mask_samples": target_masks.flatten(start_dim=1).any(dim=1),
        "empty_mask_samples": ~target_masks.flatten(start_dim=1).any(dim=1),
    }
    grouped_records: dict[str, list[dict[str, object]]] = {name: [] for name in groups}
    best_record: dict[str, object] | None = None

    for threshold in thresholds:
        predicted_masks = probabilities >= threshold
        all_record = _threshold_group_record(
            threshold=threshold,
            predicted_masks=predicted_masks,
            target_masks=target_masks,
            sample_selector=groups["all_samples"],
            include_empty_false_positive_diagnostics=False,
        )
        grouped_records["all_samples"].append(all_record)
        dice = float(all_record["metrics"]["dice"])
        if best_record is None or dice > float(best_record["dice"]):
            best_record = {"threshold": threshold, "dice": dice}

        grouped_records["positive_mask_samples"].append(
            _threshold_group_record(
                threshold=threshold,
                predicted_masks=predicted_masks,
                target_masks=target_masks,
                sample_selector=groups["positive_mask_samples"],
                include_empty_false_positive_diagnostics=False,
            )
        )
        grouped_records["empty_mask_samples"].append(
            _threshold_group_record(
                threshold=threshold,
                predicted_masks=predicted_masks,
                target_masks=target_masks,
                sample_selector=groups["empty_mask_samples"],
                include_empty_false_positive_diagnostics=True,
            )
        )

    return {
        "thresholds": thresholds,
        "default_threshold": default_threshold,
        "best_threshold_by_dice": best_record,
        "groups": grouped_records,
    }


def _threshold_group_record(
    *,
    threshold: float,
    predicted_masks: torch.Tensor,
    target_masks: torch.Tensor,
    sample_selector: torch.Tensor,
    include_empty_false_positive_diagnostics: bool,
) -> dict[str, object]:
    selected_predictions = predicted_masks[sample_selector]
    selected_targets = target_masks[sample_selector]
    sample_count = int(selected_targets.shape[0])
    if sample_count:
        metrics: dict[str, object] = binary_segmentation_metrics(selected_predictions, selected_targets)
    else:
        metrics = {"dice": None, "iou": None, "precision": None, "recall": None}
    record: dict[str, object] = {"threshold": threshold, "sample_count": sample_count, "metrics": metrics}
    if include_empty_false_positive_diagnostics:
        false_positive_pixels = int(selected_predictions.sum().item()) if sample_count else 0
        total_pixels = int(selected_predictions.numel()) if sample_count else 0
        samples_with_false_positives = (
            int(selected_predictions.flatten(start_dim=1).any(dim=1).sum().item()) if sample_count else 0
        )
        record.update(
            {
                "false_positive_pixels": false_positive_pixels,
                "false_positive_pixel_rate": false_positive_pixels / total_pixels if total_pixels else None,
                "samples_with_false_positives": samples_with_false_positives,
            }
        )
    return record
