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
from ml_autoresearch.gvccs import GVCCSDataset, discover_gvccs_samples, deterministic_train_val_split
from ml_autoresearch.metrics import binary_segmentation_metrics
from ml_autoresearch.smoke import INPUT_SPEC, OUTPUT_SPEC, _extract_mask_logits, _import_candidate_model
from ml_autoresearch.synthetic import SyntheticContrailDataset

SYNTHETIC_FIXTURE_SEED = 20260502
SAMPLING_POLICY_SEED = 20260531
TRAIN_SAMPLES = 8
VAL_SAMPLES = 4


class TrainingError(RuntimeError):
    """Raised when Harness-owned training fails."""


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

    train_loader_factory = lambda batch_size, sampling_policy: _data_loader_for_sampling(  # noqa: E731 - local concise factory.
        SyntheticContrailDataset(TRAIN_SAMPLES, seed=SYNTHETIC_FIXTURE_SEED),
        batch_size=batch_size,
        sampling_policy=sampling_policy,
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

    train_loader_factory = lambda batch_size, sampling_policy: _data_loader_for_sampling(  # noqa: E731
        GVCCSDataset(split.train), batch_size=batch_size, sampling_policy=sampling_policy
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [start_line]

    try:
        torch.manual_seed(SYNTHETIC_FIXTURE_SEED)
        manifest = yaml.safe_load(resolved_manifest_path.read_text())
        training = manifest["training"]
        sampling_policy = manifest.get("data", {}).get("sampling_policy", "sequential")
        if training["loss"] != "bce_dice":
            raise TrainingError(f"unsupported loss: {training['loss']}")

        device = _select_training_device()
        module = _import_candidate_model(candidate_dir)
        model = module.build_model(dict(INPUT_SPEC), dict(OUTPUT_SPEC))
        if not isinstance(model, torch.nn.Module):
            raise TrainingError("build_model must return a torch.nn.Module")
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
        batch_size = int(training["batch_size"])
        train_loader = train_loader_factory(batch_size, sampling_policy)
        val_loader = val_loader_factory(batch_size)

        metrics_path.write_text("")
        max_epochs = int(training["max_epochs"])
        timeout_requested = False
        final: dict[str, object] = {}
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss_total = 0.0
            trained_samples = 0
            for batch_index, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = _extract_mask_logits(model(inputs))[0]
                loss = bce_dice_loss(logits, targets)
                loss.backward()
                optimizer.step()
                trained_samples += int(inputs.shape[0])
                train_loss_total += float(loss.item()) * inputs.shape[0]
                _append_jsonl(metrics_path, {"split": "train", "epoch": epoch, "batch": batch_index, "loss": float(loss.item())})
                if _timeout_requested():
                    timeout_requested = True
                    lines.append("Wall-clock timeout requested by Harness; stopping at end-of-batch checkpoint.")
                    break

            final = _evaluate(model, val_loader, device=device)
            final["epoch"] = epoch
            final["hardware/device"] = device.type
            final["train/loss"] = train_loss_total / max(trained_samples, 1)
            if timeout_requested:
                final["run/timeout_requested"] = True
            _append_jsonl(metrics_path, {"split": "val", **final})
            if timeout_requested:
                break

        final["artifacts"] = write_prediction_sample_artifacts(
            run_dir=artifact_run_dir,
            model=model,
            data_loader=val_loader,
            split="val",
            max_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        )
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


def _data_loader_for_sampling(dataset, *, batch_size: int, sampling_policy: str) -> DataLoader:
    if sampling_policy == "sequential":
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if sampling_policy == "deterministic_shuffle":
        generator = torch.Generator()
        generator.manual_seed(SAMPLING_POLICY_SEED)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    raise TrainingError(f"unsupported sampling policy: {sampling_policy}")


def bce_dice_loss(mask_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
    probabilities = torch.sigmoid(mask_logits)
    intersection = (probabilities * target_mask).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1e-7) / (denominator + 1e-7)).mean()
    return bce + dice_loss


def _evaluate(model: torch.nn.Module, val_loader: DataLoader, *, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, target in val_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            logits = _extract_mask_logits(model(inputs))[0]
            loss = bce_dice_loss(logits, target)
            total_loss += float(loss.item()) * inputs.shape[0]
            predictions.append((torch.sigmoid(logits) >= 0.5).detach().cpu())
            targets.append((target >= 0.5).detach().cpu())
    metrics = binary_segmentation_metrics(torch.cat(predictions), torch.cat(targets))
    return {
        "val/dice": metrics["dice"],
        "val/iou": metrics["iou"],
        "val/precision": metrics["precision"],
        "val/recall": metrics["recall"],
        "val/loss": total_loss / len(val_loader.dataset),
    }


def _select_training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _timeout_requested() -> bool:
    sentinel = os.environ.get("ML_AUTORESEARCH_TIMEOUT_SENTINEL")
    return bool(sentinel and Path(sentinel).exists())
