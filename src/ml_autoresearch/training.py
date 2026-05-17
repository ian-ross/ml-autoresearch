"""Harness-owned synthetic fixture training loop."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from ml_autoresearch.artifacts import write_prediction_sample_artifacts
from ml_autoresearch.errors import TrainingError
from ml_autoresearch.gvccs import GVCCSDataset, discover_gvccs_samples, deterministic_train_val_split
from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.smoke import INPUT_SPEC, _extract_expected_outputs, _import_candidate_model, output_spec_from_resolved_manifest
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

    train_loader_factory = lambda batch_size, sampling_policy, augmentation_policy: _data_loader_for_sampling(  # noqa: E731 - local concise factory.
        SyntheticContrailDataset(TRAIN_SAMPLES, seed=SYNTHETIC_FIXTURE_SEED),
        batch_size=batch_size,
        sampling_policy=sampling_policy,
        augmentation_policy=augmentation_policy,
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
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
    )


def train_gvccs_run(
    run_dir: str | Path,
    data_root: str | Path,
    *,
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
) -> dict[str, object]:
    """Train local GVCCS RGB image and binary Contrail Mask pairs for the resolved manifest budget."""

    path = Path(run_dir)
    return train_gvccs(
        candidate_dir=path / "candidate",
        resolved_manifest_path=path / "resolved_manifest.yaml",
        outputs_dir=path / "outputs",
        artifact_run_dir=path,
        data_root=data_root,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
    )


def train_gvccs(
    *,
    candidate_dir: str | Path,
    resolved_manifest_path: str | Path,
    outputs_dir: str | Path,
    artifact_run_dir: str | Path,
    data_root: str | Path,
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
) -> dict[str, object]:
    """Train GVCCS data with explicit Harness-controlled mounted paths."""

    samples = discover_gvccs_samples(data_root, split="train", max_samples=max_samples)
    split = deterministic_train_val_split(samples)

    train_loader_factory = lambda batch_size, sampling_policy, augmentation_policy: _data_loader_for_sampling(  # noqa: E731
        GVCCSDataset(split.train), batch_size=batch_size, sampling_policy=sampling_policy, augmentation_policy=augmentation_policy
    )
    val_loader_factory = lambda batch_size: _data_loader_for_sampling(  # noqa: E731
        GVCCSDataset(split.val), batch_size=batch_size, sampling_policy="sequential"
    )
    return _train_manifest_epochs_run(
        candidate_dir=candidate_dir,
        resolved_manifest_path=resolved_manifest_path,
        outputs_dir=outputs_dir,
        artifact_run_dir=artifact_run_dir,
        start_line=f"Starting GVCCS training from {Path(data_root)} with {len(split.train)} train and {len(split.val)} val samples.",
        success_line="GVCCS training completed.",
        failure_prefix="GVCCS training failed",
        train_loader_factory=train_loader_factory,
        val_loader_factory=val_loader_factory,
        train_sample_count=len(split.train),
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
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
    max_prediction_samples: int,
    prediction_sample_policy: str,
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
        if training["loss"] != "bce_dice":
            raise TrainingError(f"unsupported loss: {training['loss']}")
        auxiliary_targets = manifest.get("auxiliary_targets", [])
        output_spec = output_spec_from_resolved_manifest(resolved_manifest_path)

        device = _select_training_device()
        module = _import_candidate_model(candidate_dir)
        model = module.build_model(dict(INPUT_SPEC), dict(output_spec))
        if not isinstance(model, torch.nn.Module):
            raise TrainingError("build_model must return a torch.nn.Module")
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
        batch_size = int(training["batch_size"])
        train_loader = train_loader_factory(batch_size, sampling_policy, augmentation_policy)
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
                logits = outputs["mask_logits"]
                mask_loss = bce_dice_loss(logits, targets)
                auxiliary_losses = _auxiliary_losses(outputs, targets, auxiliary_targets)
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

            final = _evaluate(model, val_loader, device=device, output_spec=output_spec, auxiliary_targets=auxiliary_targets)
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
            validation_value = float(validation_record["val/dice"])
            if best_validation_value is None or validation_value > best_validation_value:
                best_validation_value = validation_value
                _write_best_epoch_model_artifact(
                    best_epoch_model_path,
                    model=model,
                    epoch=epoch,
                    selection_metric="val/dice",
                    selection_value=validation_value,
                )
            _append_jsonl(metrics_path, validation_record)
            if timeout_requested:
                break

        best_metrics = _best_validation_metrics(
            validation_records, model_artifact="outputs/models/best_epoch_model.pt"
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


def _data_loader_for_sampling(dataset, *, batch_size: int, sampling_policy: str, augmentation_policy: str = "none") -> DataLoader:
    dataset = _dataset_with_augmentation_policy(dataset, augmentation_policy)
    if sampling_policy == "sequential":
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if sampling_policy == "deterministic_shuffle":
        generator = torch.Generator()
        generator.manual_seed(SAMPLING_POLICY_SEED)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    raise TrainingError(f"unsupported sampling policy: {sampling_policy}")


def _dataset_with_augmentation_policy(dataset, augmentation_policy: str):
    if augmentation_policy == "none":
        return dataset
    if augmentation_policy in {"light_geometric", "light_photometric", "light_combined"}:
        return _AugmentedContrailDataset(dataset, augmentation_policy)
    raise TrainingError(f"unsupported augmentation policy: {augmentation_policy}")


class _AugmentedContrailDataset(torch.utils.data.Dataset):
    """Harness-owned deterministic augmentation wrapper for training samples."""

    def __init__(self, dataset, augmentation_policy: str) -> None:
        self.dataset = dataset
        self.augmentation_policy = augmentation_policy

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, mask = self.dataset[index]
        if self.augmentation_policy in {"light_geometric", "light_combined"}:
            image, mask = _apply_light_geometric_augmentation(image, mask, index=index)
        if self.augmentation_policy in {"light_photometric", "light_combined"}:
            image = _apply_light_photometric_augmentation(image, index=index)
        return image, mask


def _apply_light_geometric_augmentation(
    image: torch.Tensor, mask: torch.Tensor, *, index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Horizontal mirroring is safe for GVCCS whole-sky contrail masks and keeps
    # thin line geometry image-aligned. Apply on odd indices so tests and Runs
    # remain deterministic under the Harness-owned policy.
    if index % 2 == 1:
        return torch.flip(image, dims=[2]), torch.flip(mask, dims=[2])
    return image, mask


def _apply_light_photometric_augmentation(image: torch.Tensor, *, index: int) -> torch.Tensor:
    # Conservative GVCCS-specific brightness/contrast perturbation plus a tiny
    # deterministic sensor-noise term. The Contrail Mask is intentionally not
    # changed by photometric augmentation.
    contrast = 1.05 if index % 2 == 0 else 0.95
    brightness = 0.025 if index % 3 == 0 else -0.015
    adjusted = (image - 0.5) * contrast + 0.5 + brightness
    generator = torch.Generator(device=image.device).manual_seed(SYNTHETIC_FIXTURE_SEED + int(index))
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype) * 0.005
    return torch.clamp(adjusted + noise, 0.0, 1.0)


def bce_dice_loss(mask_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
    probabilities = torch.sigmoid(mask_logits)
    intersection = (probabilities * target_mask).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1e-7) / (denominator + 1e-7)).mean()
    return bce + dice_loss


def derive_line_target_v1(target_mask: torch.Tensor) -> torch.Tensor:
    """Derive the v1 Harness-owned Line Target as a small tolerance band around positives."""

    return F.max_pool2d(target_mask.float(), kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive_weight = torch.tensor(4.0, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)


def _auxiliary_losses(
    outputs: dict[str, torch.Tensor], target_mask: torch.Tensor, auxiliary_targets: list[dict[str, object]]
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for target in auxiliary_targets:
        if target.get("name") != "line" or target.get("loss") != "weighted_bce":
            raise TrainingError(f"unsupported auxiliary target in resolved manifest: {target}")
        line_target = derive_line_target_v1(target_mask)
        weight = float(target["weight"])
        losses["line"] = weight * weighted_bce_loss(outputs[str(target["output"])], line_target)
    return losses


def _evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    *,
    device: torch.device,
    output_spec: dict[str, object] | None = None,
    auxiliary_targets: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    model.eval()
    total_mask_loss = 0.0
    total_loss = 0.0
    auxiliary_loss_totals: dict[str, float] = {}
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    output_spec = output_spec or {"form": "mask_logits", "shape": [1, 128, 128]}
    auxiliary_targets = auxiliary_targets or []
    with torch.no_grad():
        for inputs, target in val_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            outputs = _extract_expected_outputs(model(inputs), output_spec)
            logits = outputs["mask_logits"]
            mask_loss = bce_dice_loss(logits, target)
            auxiliary_losses = _auxiliary_losses(outputs, target, auxiliary_targets)
            batch_total_loss = mask_loss + sum(auxiliary_losses.values())
            total_mask_loss += float(mask_loss.item()) * inputs.shape[0]
            total_loss += float(batch_total_loss.item()) * inputs.shape[0]
            for name, auxiliary_loss in auxiliary_losses.items():
                auxiliary_loss_totals[name] = auxiliary_loss_totals.get(name, 0.0) + float(auxiliary_loss.item()) * inputs.shape[0]
            predictions.append((torch.sigmoid(logits) >= 0.5).detach().cpu())
            targets.append((target >= 0.5).detach().cpu())
    metrics = binary_segmentation_metrics(torch.cat(predictions), torch.cat(targets))
    sample_count = len(val_loader.dataset)
    result = {
        "val/dice": metrics["dice"],
        "val/iou": metrics["iou"],
        "val/precision": metrics["precision"],
        "val/recall": metrics["recall"],
        "val/loss": total_mask_loss / sample_count,
    }
    if auxiliary_loss_totals:
        for name, total in auxiliary_loss_totals.items():
            result[f"val/aux/{name}_loss"] = total / sample_count
        result["val/total_loss"] = total_loss / sample_count
    return result


def _best_validation_metrics(
    validation_records: list[dict[str, object]], *, model_artifact: str | None = None
) -> dict[str, object]:
    if not validation_records:
        raise TrainingError("cannot report best validation metrics without validation records")
    metric_name = "val/dice"
    selected = max(validation_records, key=lambda record: float(record[metric_name]))
    metrics = {key: value for key, value in selected.items() if key != "split"}
    summary = {
        "epoch": selected["epoch"],
        "selection_metric": metric_name,
        "selection_mode": "max",
        "selection_value": selected[metric_name],
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
