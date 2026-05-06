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

from ml_autoresearch.gvccs import GVCCSDataset, discover_gvccs_samples, deterministic_train_val_split
from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.smoke import INPUT_SPEC, OUTPUT_SPEC, _extract_mask_logits, _import_candidate_model

DEFAULT_EVALUATION_THRESHOLD = 0.5


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

        aggregate, per_sample_records = _evaluate_gvccs_validation_split(
            run_dir=source_run_dir,
            data_root=resolved_data_root,
            model_artifact_path=model_artifact_path,
            threshold=threshold,
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

        completed_metadata = {
            **base_metadata,
            "status": "completed",
            "completed_at": _now_iso(),
            "data_root": str(resolved_data_root),
            "model_artifact": model_artifact,
            "artifacts": {
                "aggregate_metrics": "aggregate_metrics.json",
                "per_sample_metrics": "per_sample_metrics.jsonl",
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
) -> tuple[dict[str, float], list[dict[str, object]]]:
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

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    per_sample_records: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch_indices = list(range(start, min(start + batch_size, len(dataset))))
            inputs = torch.stack([dataset[index][0] for index in batch_indices]).to(device)
            targets = torch.stack([dataset[index][1] for index in batch_indices]).to(device)
            logits = _extract_mask_logits(model(inputs))[0]
            predictions = (torch.sigmoid(logits) >= threshold).detach().cpu()
            target_masks = (targets >= 0.5).detach().cpu()
            all_predictions.append(predictions)
            all_targets.append(target_masks)
            for offset, index in enumerate(batch_indices):
                metrics = binary_segmentation_metrics(predictions[offset : offset + 1], target_masks[offset : offset + 1])
                sample = dataset.samples[index]
                per_sample_records.append(
                    {
                        "sample_id": f"val/{index:06d}",
                        "image_id": sample.image_id,
                        "image_path": str(sample.image_path),
                        **metrics,
                    }
                )
    aggregate = binary_segmentation_metrics(torch.cat(all_predictions), torch.cat(all_targets))
    return aggregate, per_sample_records


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
