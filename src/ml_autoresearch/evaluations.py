"""Harness-owned Post-Run Evaluation operations."""

from __future__ import annotations

import json
import secrets
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
import yaml

from ml_autoresearch.artifacts import _save_mask_tensor, _save_overlay, _save_probability_heatmap, _save_rgb_tensor
from ml_autoresearch.gvccs import GVCCSDataset, discover_gvccs_samples, deterministic_train_val_split
from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.smoke import INPUT_SPEC, OUTPUT_SPEC, _extract_mask_logits, _import_candidate_model

DEFAULT_EVALUATION_THRESHOLD = 0.5
DEFAULT_MAX_ARTIFACT_SAMPLES = 12


class EvaluationError(RuntimeError):
    """Raised when a Post-Run Evaluation fails clearly."""


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_id: str
    evaluation_dir: Path
    status: Literal["completed", "failed"]
    failure_reason: str | None = None


def evaluate_run(
    run_dir: str | Path,
    *,
    split: Literal["val"] = "val",
    backend: Literal["native"] = "native",
    data_root: str | Path | None = None,
    threshold: float = DEFAULT_EVALUATION_THRESHOLD,
    max_artifact_samples: int = DEFAULT_MAX_ARTIFACT_SAMPLES,
) -> EvaluationResult:
    """Run native Whole-Validation Failure Analysis for a completed Run without retraining."""

    source_run_dir = Path(run_dir)
    evaluation_id = _generate_evaluation_id(source_run_dir)
    evaluation_dir = source_run_dir / "outputs" / "evaluations" / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=False)

    started_at = _now_iso()
    base_metadata = _base_metadata(
        evaluation_id=evaluation_id,
        evaluation_dir=evaluation_dir,
        run_dir=source_run_dir,
        split=split,
        backend=backend,
        threshold=threshold,
        started_at=started_at,
    )
    _write_metadata(evaluation_dir, {**base_metadata, "status": "running"})

    try:
        if backend != "native":
            raise EvaluationError("evaluate-run currently supports only --backend native")
        if split != "val":
            raise EvaluationError("evaluate-run currently supports only --split val")
        metadata = _read_json(source_run_dir / "run_metadata.json")
        if metadata.get("status") != "completed":
            raise EvaluationError(f"source Run must be completed, got status: {metadata.get('status')}")
        resolved_data_root = _resolve_data_root(metadata, data_root)
        model_artifact = _model_artifact_from_best_metrics(source_run_dir)
        model_artifact_path = source_run_dir / model_artifact
        if not model_artifact_path.is_file():
            raise EvaluationError(f"model artifact is missing: {model_artifact}")

        if max_artifact_samples < 1:
            raise EvaluationError("max_artifact_samples must be at least 1")
        aggregate, per_sample_records, threshold_sweep, diagnostic_manifest = _evaluate_gvccs_validation_split(
            run_dir=source_run_dir,
            data_root=resolved_data_root,
            model_artifact_path=model_artifact_path,
            threshold=threshold,
            evaluation_dir=evaluation_dir,
            max_artifact_samples=max_artifact_samples,
        )
        aggregate_payload = {
            "evaluation_id": evaluation_id,
            "split": split,
            "threshold": threshold,
            "sample_count": len(per_sample_records),
            "metrics": aggregate,
        }
        (evaluation_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n")
        with (evaluation_dir / "per_sample_metrics.jsonl").open("w") as handle:
            for record in per_sample_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        (evaluation_dir / "threshold_sweep.json").write_text(json.dumps(threshold_sweep, indent=2, sort_keys=True) + "\n")
        diagnostics_dir = evaluation_dir / "diagnostic_samples"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (diagnostics_dir / "samples.json").write_text(json.dumps(diagnostic_manifest, indent=2, sort_keys=True) + "\n")

        completed_metadata = {
            **base_metadata,
            "status": "completed",
            "completed_at": _now_iso(),
            "data_root": str(resolved_data_root),
            "model_artifact": model_artifact,
            "artifacts": {
                "aggregate_metrics": "aggregate_metrics.json",
                "per_sample_metrics": "per_sample_metrics.jsonl",
                "threshold_sweep": "threshold_sweep.json",
                "diagnostic_samples": "diagnostic_samples/samples.json",
            },
        }
        _write_metadata(evaluation_dir, completed_metadata)
        return EvaluationResult(evaluation_id, evaluation_dir, "completed")
    except Exception as exc:  # noqa: BLE001 - failed evaluation metadata must capture unexpected failures too.
        reason = str(exc)
        failed_metadata = {
            **base_metadata,
            "status": "failed",
            "failed_at": _now_iso(),
            "failure_reason": reason,
            "traceback": traceback.format_exc(),
        }
        _write_metadata(evaluation_dir, failed_metadata)
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError(reason) from exc


def _evaluate_gvccs_validation_split(
    *,
    run_dir: Path,
    data_root: Path,
    model_artifact_path: Path,
    threshold: float,
    evaluation_dir: Path,
    max_artifact_samples: int,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, object], dict[str, object]]:
    manifest = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    batch_size = int(manifest.get("training", {}).get("batch_size", 1))
    samples = discover_gvccs_samples(data_root, split="train")
    val_samples = deterministic_train_val_split(samples).val
    dataset = GVCCSDataset(val_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = _import_candidate_model(run_dir / "candidate")
    model = module.build_model(dict(INPUT_SPEC), dict(OUTPUT_SPEC))
    if not isinstance(model, torch.nn.Module):
        raise EvaluationError("build_model must return a torch.nn.Module")
    checkpoint = torch.load(model_artifact_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise EvaluationError(f"model artifact is unreadable: missing model_state_dict in {model_artifact_path}")
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
            logits = _extract_mask_logits(model(inputs))[0]
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
                per_sample_records.append(
                    {
                        "sample_id": f"val/{index:06d}",
                        "dataset_index": index,
                        "image_id": sample.image_id,
                        "image_path": str(sample.image_path),
                        **metrics,
                        **_confusion_counts(sample_prediction, sample_target),
                    }
                )
    input_tensor = torch.cat(all_inputs)
    prediction_tensor = torch.cat(all_predictions)
    probability_tensor = torch.cat(all_probabilities)
    target_tensor = torch.cat(all_targets)
    aggregate = binary_segmentation_metrics(prediction_tensor, target_tensor)
    threshold_sweep = _threshold_sweep_summary(probability_tensor, target_tensor)
    diagnostic_manifest = _write_diagnostic_sample_artifacts(
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


def _confusion_counts(predicted_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, int]:
    pred = predicted_mask.bool()
    target = target_mask.bool()
    return {
        "positive_pixel_count": int(target.sum().item()),
        "predicted_positive_pixel_count": int(pred.sum().item()),
        "true_positive_pixels": int(torch.logical_and(pred, target).sum().item()),
        "false_positive_pixels": int(torch.logical_and(pred, ~target).sum().item()),
        "false_negative_pixels": int(torch.logical_and(~pred, target).sum().item()),
    }


def _write_diagnostic_sample_artifacts(
    *,
    evaluation_dir: Path,
    dataset: GVCCSDataset,
    inputs: torch.Tensor,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    per_sample_records: list[dict[str, object]],
    threshold: float,
    max_artifact_samples: int,
) -> dict[str, object]:
    diagnostics_dir = evaluation_dir / "diagnostic_samples"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    selections = _select_failure_bucket_indices(per_sample_records, max_artifact_samples=max_artifact_samples)
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
        image = inputs[dataset_index].detach().cpu().clamp(0.0, 1.0)
        target = targets[dataset_index].detach().cpu().bool()
        prediction = predictions[dataset_index].detach().cpu().bool()
        probability = probabilities[dataset_index].detach().cpu()
        _save_rgb_tensor(diagnostics_dir / paths["input"], image)
        _save_mask_tensor(diagnostics_dir / paths["ground_truth"], target)
        _save_mask_tensor(diagnostics_dir / paths["prediction"], prediction)
        _save_overlay(diagnostics_dir / paths["overlay"], image, target, prediction)
        _save_probability_heatmap(diagnostics_dir / paths["probability_heatmap"], probability)

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


def _select_failure_bucket_indices(
    per_sample_records: list[dict[str, object]], *, max_artifact_samples: int
) -> list[dict[str, object]]:
    if max_artifact_samples < 1:
        raise EvaluationError("max_artifact_samples must be at least 1")

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


def _threshold_sweep_summary(probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, object]:
    """Compute aggregate threshold diagnostics for Whole-Validation Failure Analysis."""

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
        "default_threshold": DEFAULT_EVALUATION_THRESHOLD,
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
        metrics = binary_segmentation_metrics(selected_predictions, selected_targets)
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


def _model_artifact_from_best_metrics(run_dir: Path) -> str:
    best_metrics_path = run_dir / "outputs" / "best_metrics.json"
    if not best_metrics_path.is_file():
        raise EvaluationError("best metrics artifact is missing: outputs/best_metrics.json")
    best_metrics = _read_json(best_metrics_path)
    model_artifact = best_metrics.get("model_artifact")
    if not isinstance(model_artifact, str) or not model_artifact:
        raise EvaluationError("best metrics artifact does not contain model_artifact")
    return model_artifact


def _resolve_data_root(metadata: dict[str, object], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        raise EvaluationError("source Run metadata does not contain GVCCS data root; pass --data-root")
    host_data_path = dataset.get("host_data_path")
    if not isinstance(host_data_path, str) or not host_data_path:
        raise EvaluationError("source Run metadata does not contain GVCCS data root; pass --data-root")
    return Path(host_data_path)


def _base_metadata(
    *,
    evaluation_id: str,
    evaluation_dir: Path,
    run_dir: Path,
    split: str,
    backend: str,
    threshold: float,
    started_at: str,
) -> dict[str, object]:
    return {
        "evaluation_id": evaluation_id,
        "mode": "whole_validation_failure_analysis",
        "status": "running",
        "split": split,
        "backend": backend,
        "threshold": threshold,
        "started_at": started_at,
        "evaluation_dir": str(evaluation_dir),
        "source_run": {"run_id": run_dir.name, "run_dir": str(run_dir)},
        "reserved_statuses": ["running", "completed", "failed"],
    }


def _generate_evaluation_id(run_dir: Path) -> str:
    evaluations_dir = run_dir / "outputs" / "evaluations"
    while True:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)
        evaluation_id = f"eval_{timestamp}_{suffix}"
        if not (evaluations_dir / evaluation_id).exists():
            return evaluation_id


def _write_metadata(evaluation_dir: Path, payload: dict[str, object]) -> None:
    (evaluation_dir / "evaluation_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise EvaluationError(f"required artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"required artifact is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"required artifact must contain a JSON object: {path}")
    return payload


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
