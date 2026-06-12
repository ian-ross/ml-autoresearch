"""Harness-owned synthetic fixture training loop."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from ml_autoresearch.artifacts import write_prediction_sample_artifacts
from ml_autoresearch.errors import TrainingError
from ml_autoresearch.research_problems import ResearchProblemProviderConfig, load_research_problem_provider
from ml_autoresearch.training_adapters import ResearchProblemTrainingAdapter
from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.problem_support.segmentation import bce_dice_loss, derive_boundary_target_v1
from ml_autoresearch.smoke import _extract_expected_outputs, _import_candidate_model, input_spec_from_resolved_manifest, output_spec_from_resolved_manifest
from ml_autoresearch.synthetic import SyntheticContrailDataset

SYNTHETIC_FIXTURE_SEED = 20260502
SAMPLING_POLICY_SEED = 20260531
TRAIN_SAMPLES = 8
VAL_SAMPLES = 4


def train_synthetic_fixture_run(
    run_dir: str | Path, *, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n"
) -> dict[str, object]:
    """Train deterministic generated Contrail Mask fixture data for the resolved manifest budget."""

    path = Path(run_dir)
    return train_synthetic_fixture(
        candidate_dir=path / "candidate",
        resolved_manifest_path=path / "resolved_manifest.yaml",
        outputs_dir=path / "outputs",
        artifact_run_dir=path,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
    )


def train_synthetic_fixture(
    *,
    candidate_dir: str | Path,
    resolved_manifest_path: str | Path,
    outputs_dir: str | Path,
    artifact_run_dir: str | Path,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
) -> dict[str, object]:
    """Train deterministic generated Contrail Mask fixture data with explicit mounted paths."""

    training_adapter = None
    train_loader_factory = lambda batch_size, sampling_policy, augmentation_policy: _data_loader_for_sampling(  # noqa: E731 - local concise factory.
        SyntheticContrailDataset(TRAIN_SAMPLES, seed=SYNTHETIC_FIXTURE_SEED),
        batch_size=batch_size,
        sampling_policy=sampling_policy,
        augmentation_policy=augmentation_policy,
        training_adapter=training_adapter,
    )
    val_loader_factory = lambda batch_size: _data_loader_for_sampling(  # noqa: E731
        SyntheticContrailDataset(VAL_SAMPLES, seed=SYNTHETIC_FIXTURE_SEED + 10_000),
        batch_size=batch_size,
        sampling_policy="sequential",
    )
    return _train_manifest_epochs_run(
        candidate_dir=candidate_dir,
        resolved_manifest_path=resolved_manifest_path,
        outputs_dir=outputs_dir,
        artifact_run_dir=artifact_run_dir,
        start_line="Starting deterministic synthetic contrail fixture training.",
        success_line="Synthetic training completed.",
        failure_prefix="Synthetic training failed",
        train_loader_factory=train_loader_factory,
        val_loader_factory=val_loader_factory,
        train_sample_count=TRAIN_SAMPLES,
        val_sample_count=VAL_SAMPLES,
        data_policy_metadata={},
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
        training_adapter=training_adapter,
    )


def train_research_problem_run(
    run_dir: str | Path,
    provider_config: ResearchProblemProviderConfig,
    *,
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
) -> dict[str, object]:
    """Train a Run through the active trusted Research Problem Spec adapter."""

    loaded = load_research_problem_provider(provider_config)
    adapter = loaded.spec.training_adapter
    if adapter is None:
        raise TrainingError(f"Research Problem {loaded.spec.id!r} does not provide a training adapter")
    path = Path(run_dir)
    return train_research_problem(
        candidate_dir=path / "candidate",
        resolved_manifest_path=path / "resolved_manifest.yaml",
        outputs_dir=path / "outputs",
        artifact_run_dir=path,
        training_adapter=adapter,
        data_config=provider_config.data_config,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
    )


def train_research_problem(
    *,
    candidate_dir: str | Path,
    resolved_manifest_path: str | Path,
    outputs_dir: str | Path,
    artifact_run_dir: str | Path,
    training_adapter: ResearchProblemTrainingAdapter,
    data_config: dict[str, object],
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
) -> dict[str, object]:
    """Train with explicit Harness-controlled paths via a Research Problem adapter."""

    try:
        datasets = training_adapter.build_datasets(
            data_config=data_config,
            resolved_manifest_path=resolved_manifest_path,
            max_samples=max_samples,
        )
    except TrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - trusted adapter failures should fail the Run clearly.
        raise TrainingError(str(exc)) from exc
    train_loader_factory = lambda batch_size, sampling_policy, augmentation_policy: _data_loader_for_sampling(  # noqa: E731
        datasets.train_dataset,
        batch_size=batch_size,
        sampling_policy=sampling_policy,
        augmentation_policy=augmentation_policy,
        training_adapter=training_adapter,
    )
    val_loader_factory = lambda batch_size: _data_loader_for_sampling(  # noqa: E731
        datasets.validation_dataset, batch_size=batch_size, sampling_policy="sequential"
    )
    return _train_manifest_epochs_run(
        candidate_dir=candidate_dir,
        resolved_manifest_path=resolved_manifest_path,
        outputs_dir=outputs_dir,
        artifact_run_dir=artifact_run_dir,
        start_line=datasets.start_line,
        success_line=datasets.success_line,
        failure_prefix=datasets.failure_prefix,
        train_loader_factory=train_loader_factory,
        val_loader_factory=val_loader_factory,
        train_sample_count=len(datasets.train_dataset),
        val_sample_count=len(datasets.validation_dataset),
        data_policy_metadata=datasets.data_policy_metadata,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
        training_adapter=training_adapter,
    )


def _train_manifest_epochs_run(
    *,
    candidate_dir: str | Path,
    resolved_manifest_path: str | Path,
    outputs_dir: str | Path,
    artifact_run_dir: str | Path,
    start_line: str,
    success_line: str,
    failure_prefix: str,
    train_loader_factory,
    val_loader_factory,
    train_sample_count: int,
    val_sample_count: int,
    data_policy_metadata: dict[str, object],
    max_prediction_samples: int,
    prediction_sample_policy: str,
    training_adapter: ResearchProblemTrainingAdapter | object | None = None,
) -> dict[str, object]:
    candidate_dir = Path(candidate_dir)
    resolved_manifest_path = Path(resolved_manifest_path)
    outputs_dir = Path(outputs_dir)
    artifact_run_dir = Path(artifact_run_dir)
    log_path = outputs_dir / "logs" / "training.log"
    metrics_path = outputs_dir / "metrics.jsonl"
    final_metrics_path = outputs_dir / "final_metrics.json"
    best_metrics_path = outputs_dir / "best_metrics.json"
    best_epoch_model_path = outputs_dir / "models" / "best_epoch_model.pt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [start_line]

    try:
        torch.manual_seed(SYNTHETIC_FIXTURE_SEED)
        manifest = yaml.safe_load(resolved_manifest_path.read_text())
        training = manifest["training"]
        data_policy = manifest.get("data", {})
        sampling_policy = data_policy.get("sampling_policy", "sequential")
        augmentation_policy = data_policy.get("augmentation_policy_effective", data_policy.get("augmentation_policy", "none"))
        primary_loss_name = str(training["loss"])
        auxiliary_targets = manifest.get("auxiliary_targets", [])
        input_spec = input_spec_from_resolved_manifest(resolved_manifest_path)
        output_spec = output_spec_from_resolved_manifest(resolved_manifest_path)

        device = _select_training_device()
        module = _import_candidate_model(candidate_dir)
        model = module.build_model(dict(input_spec), dict(output_spec))
        if not isinstance(model, torch.nn.Module):
            raise TrainingError("build_model must return a torch.nn.Module")
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
        batch_size = int(training["batch_size"])
        train_loader = train_loader_factory(batch_size, sampling_policy, augmentation_policy)
        primary_output_name = _primary_output_name(output_spec, training_adapter)
        selection_metric, selection_mode = _selection_policy(training_adapter)
        val_loader = val_loader_factory(batch_size)

        metrics_path.write_text("")
        max_epochs = int(training["max_epochs"])
        timeout_requested = False
        final: dict[str, object] = {}
        validation_records: list[dict[str, object]] = []
        best_validation_value: float | None = None
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_mask_loss_total = 0.0
            train_aux_loss_totals: dict[str, float] = {}
            trained_samples = 0
            for batch_index, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = _extract_expected_outputs(model(inputs), output_spec)
                logits = outputs[primary_output_name]
                mask_loss = _primary_loss(primary_loss_name, logits, targets, training_adapter)
                auxiliary_losses = _auxiliary_losses(outputs, targets, auxiliary_targets, training_adapter)
                loss = mask_loss + sum(auxiliary_losses.values())
                loss.backward()
                optimizer.step()
                trained_samples += int(inputs.shape[0])
                train_loss_total += float(loss.item()) * inputs.shape[0]
                train_mask_loss_total += float(mask_loss.item()) * inputs.shape[0]
                for name, auxiliary_loss in auxiliary_losses.items():
                    train_aux_loss_totals[name] = train_aux_loss_totals.get(name, 0.0) + float(auxiliary_loss.item()) * inputs.shape[0]
                batch_record = {
                    "split": "train",
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": float(loss.item()),
                    "mask_loss": float(mask_loss.item()),
                }
                batch_record.update({f"aux/{name}_loss": float(value.item()) for name, value in auxiliary_losses.items()})
                _append_jsonl(metrics_path, batch_record)
                if _timeout_requested():
                    timeout_requested = True
                    lines.append("Wall-clock timeout requested by Harness; stopping at end-of-batch checkpoint.")
                    break

            final = _evaluate(
                model,
                val_loader,
                device=device,
                output_spec=output_spec,
                auxiliary_targets=auxiliary_targets,
                primary_loss_name=primary_loss_name,
                training_adapter=training_adapter,
            )
            final["epoch"] = epoch
            final["hardware/device"] = device.type
            final["train/loss"] = train_loss_total / max(trained_samples, 1)
            final["train/mask_loss"] = train_mask_loss_total / max(trained_samples, 1)
            for name, total in train_aux_loss_totals.items():
                final[f"train/aux/{name}_loss"] = total / max(trained_samples, 1)
            if timeout_requested:
                final["run/timeout_requested"] = True
            validation_record = {"split": "val", **final}
            validation_records.append(dict(validation_record))
            validation_value = float(validation_record[selection_metric])
            is_better = validation_value > best_validation_value if selection_mode == "max" and best_validation_value is not None else validation_value < best_validation_value if best_validation_value is not None else True
            if is_better:
                best_validation_value = validation_value
                _write_best_epoch_model_artifact(
                    best_epoch_model_path,
                    model=model,
                    epoch=epoch,
                    selection_metric=selection_metric,
                    selection_value=validation_value,
                )
            _append_jsonl(metrics_path, validation_record)
            if timeout_requested:
                break

        final["sample_counts"] = {"train": train_sample_count, "validation": val_sample_count}
        if data_policy_metadata:
            final["data_policy"] = data_policy_metadata
        best_metrics = _best_validation_metrics(
            validation_records,
            model_artifact="outputs/models/best_epoch_model.pt",
            selection_metric=selection_metric,
            selection_mode=selection_mode,
        )
        best_metrics_path.write_text(json.dumps(best_metrics, indent=2, sort_keys=True) + "\n")
        artifacts = write_prediction_sample_artifacts(
            run_dir=artifact_run_dir,
            model=model,
            data_loader=val_loader,
            split="val",
            max_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
            output_spec=output_spec,
            sample_selector=_prediction_sample_selector(training_adapter, prediction_sample_policy, max_prediction_samples),
            display_input_renderer=_prediction_sample_input_renderer(training_adapter),
            sample_artifact_writer=_prediction_sample_artifact_writer(training_adapter),
        )
        artifacts["best_metrics"] = "outputs/best_metrics.json"
        artifacts["best_epoch_model"] = "outputs/models/best_epoch_model.pt"
        final["artifacts"] = artifacts
        final_metrics_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
        if timeout_requested:
            lines.append("Training exited cleanly after Harness timeout request.")
        else:
            lines.append(success_line)
        log_path.write_text("\n".join(lines) + "\n")
        return final
    except Exception as exc:  # noqa: BLE001 - persist clear Harness failure details.
        reason = str(exc)
        lines.append(f"{failure_prefix}: {reason}")
        lines.append(traceback.format_exc())
        log_path.write_text("\n".join(lines) + "\n")
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(reason) from exc


def _prediction_sample_selector(training_adapter: object | None, policy: str, max_samples: int):
    selector = getattr(training_adapter, "select_prediction_samples", None)
    if selector is None:
        return None

    def select(dataset: object) -> list[dict[str, object]]:
        return selector(dataset, policy=policy, max_samples=max_samples)

    return select


def _prediction_sample_input_renderer(training_adapter: object | None):
    renderer = getattr(training_adapter, "display_prediction_sample_input", None)
    if renderer is None:
        return None
    return renderer


def _prediction_sample_artifact_writer(training_adapter: object | None):
    writer = getattr(training_adapter, "write_prediction_sample_images", None)
    if writer is None:
        return None
    return writer


def _data_loader_for_sampling(
    dataset,
    *,
    batch_size: int,
    sampling_policy: str,
    augmentation_policy: str = "none",
    training_adapter: ResearchProblemTrainingAdapter | object | None = None,
) -> DataLoader:
    dataset = _dataset_with_augmentation_policy(dataset, augmentation_policy, training_adapter)
    if sampling_policy == "sequential":
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if sampling_policy == "deterministic_shuffle":
        generator = torch.Generator()
        generator.manual_seed(SAMPLING_POLICY_SEED)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    raise TrainingError(f"unsupported sampling policy: {sampling_policy}")


def _dataset_with_augmentation_policy(dataset, augmentation_policy: str, training_adapter: ResearchProblemTrainingAdapter | object | None = None):
    if training_adapter is not None and hasattr(training_adapter, "apply_augmentation_policy"):
        return training_adapter.apply_augmentation_policy(dataset, augmentation_policy)
    if augmentation_policy == "none":
        return dataset
    if augmentation_policy in {"light_geometric", "light_photometric", "light_combined"}:
        return _LegacyLightAugmentedDataset(dataset, augmentation_policy)
    raise TrainingError(f"unsupported augmentation policy: {augmentation_policy}")


class _LegacyLightAugmentedDataset:
    """Compatibility wrapper for old synthetic fixture augmentation tests."""

    def __init__(self, dataset, augmentation_policy: str) -> None:
        self.dataset = dataset
        self.augmentation_policy = augmentation_policy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        if self.augmentation_policy in {"light_geometric", "light_combined"}:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if self.augmentation_policy in {"light_photometric", "light_combined"}:
            image = torch.clamp(image * 0.97 + 0.015, 0.0, 1.0)
        return image, mask


def _primary_output_name(output_spec: dict[str, object], training_adapter: ResearchProblemTrainingAdapter | object | None = None) -> str:
    if training_adapter is not None and hasattr(training_adapter, "primary_output_name"):
        return str(training_adapter.primary_output_name(output_spec))
    return str(output_spec.get("form"))


def _primary_loss(
    loss_name: str,
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    training_adapter: ResearchProblemTrainingAdapter | object | None = None,
) -> torch.Tensor:
    if training_adapter is not None and hasattr(training_adapter, "compute_primary_loss"):
        return training_adapter.compute_primary_loss(loss_name, logits, target_mask)
    if loss_name != "bce_dice":
        raise TrainingError(f"unsupported loss: {loss_name}")
    return bce_dice_loss(logits, target_mask)


def _auxiliary_losses(
    outputs: dict[str, torch.Tensor],
    target_mask: torch.Tensor,
    auxiliary_targets: list[dict[str, object]],
    training_adapter: ResearchProblemTrainingAdapter | object | None = None,
) -> dict[str, torch.Tensor]:
    if training_adapter is not None and hasattr(training_adapter, "compute_auxiliary_losses"):
        return training_adapter.compute_auxiliary_losses(outputs, target_mask, auxiliary_targets)
    if auxiliary_targets:
        raise TrainingError("auxiliary targets require Research Problem training adapter support")
    return {}


def _validation_metrics(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    training_adapter: ResearchProblemTrainingAdapter | object | None = None,
) -> dict[str, float]:
    if training_adapter is not None and hasattr(training_adapter, "compute_validation_metrics"):
        return training_adapter.compute_validation_metrics(logits, target_mask)
    metrics = binary_segmentation_metrics(torch.sigmoid(logits) >= 0.5, target_mask >= 0.5)
    return {
        "val/dice": metrics["dice"],
        "val/iou": metrics["iou"],
        "val/precision": metrics["precision"],
        "val/recall": metrics["recall"],
    }


def _selection_policy(training_adapter: ResearchProblemTrainingAdapter | object | None = None) -> tuple[str, str]:
    if training_adapter is not None and hasattr(training_adapter, "selection_policy"):
        metric, mode = training_adapter.selection_policy()
        if mode not in {"max", "min"}:
            raise TrainingError(f"unsupported selection mode: {mode}")
        return str(metric), str(mode)
    return "val/dice", "max"


def _evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    *,
    device: torch.device,
    output_spec: dict[str, object] | None = None,
    auxiliary_targets: list[dict[str, object]] | None = None,
    primary_loss_name: str = "bce_dice",
    training_adapter: ResearchProblemTrainingAdapter | object | None = None,
) -> dict[str, float]:
    model.eval()
    total_mask_loss = 0.0
    total_loss = 0.0
    auxiliary_loss_totals: dict[str, float] = {}
    prediction_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    output_spec = output_spec or {"form": "mask_logits", "shape": [1, 128, 128]}
    auxiliary_targets = auxiliary_targets or []
    primary_output_name = _primary_output_name(output_spec, training_adapter)
    with torch.no_grad():
        for inputs, target in val_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = _extract_expected_outputs(model(inputs), output_spec)
            logits = outputs[primary_output_name]
            mask_loss = _primary_loss(primary_loss_name, logits, target, training_adapter)
            auxiliary_losses = _auxiliary_losses(outputs, target, auxiliary_targets, training_adapter)
            batch_total_loss = mask_loss + sum(auxiliary_losses.values())
            total_mask_loss += float(mask_loss.item()) * inputs.shape[0]
            total_loss += float(batch_total_loss.item()) * inputs.shape[0]
            for name, auxiliary_loss in auxiliary_losses.items():
                auxiliary_loss_totals[name] = auxiliary_loss_totals.get(name, 0.0) + float(auxiliary_loss.item()) * inputs.shape[0]
            prediction_logits.append(logits.detach().cpu())
            targets.append(target.detach().cpu())
    sample_count = len(val_loader.dataset)
    result = _validation_metrics(torch.cat(prediction_logits), torch.cat(targets), training_adapter)
    result["val/loss"] = total_mask_loss / sample_count
    if auxiliary_loss_totals:
        for name, total in auxiliary_loss_totals.items():
            result[f"val/aux/{name}_loss"] = total / sample_count
        result["val/total_loss"] = total_loss / sample_count
    return result


def _best_validation_metrics(
    validation_records: list[dict[str, object]],
    *,
    model_artifact: str | None = None,
    selection_metric: str = "val/dice",
    selection_mode: str = "max",
) -> dict[str, object]:
    if not validation_records:
        raise TrainingError("cannot report best validation metrics without validation records")
    selector = max if selection_mode == "max" else min
    selected = selector(validation_records, key=lambda record: float(record[selection_metric]))
    metrics = {key: value for key, value in selected.items() if key != "split"}
    summary = {
        "epoch": selected["epoch"],
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selection_value": selected[selection_metric],
        "metrics": metrics,
    }
    if model_artifact is not None:
        summary["model_artifact"] = model_artifact
    return summary


def _write_best_epoch_model_artifact(
    path: Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    selection_metric: str,
    selection_value: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_type": "best_epoch_model",
            "epoch": epoch,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "model_state_dict": _cpu_state_dict(model),
        },
        path,
    )


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _select_training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _timeout_requested() -> bool:
    sentinel = os.environ.get("ML_AUTORESEARCH_TIMEOUT_SENTINEL")
    return bool(sentinel and Path(sentinel).exists())
