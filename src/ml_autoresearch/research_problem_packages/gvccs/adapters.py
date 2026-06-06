"""Trusted adapters for the Ground-Camera Contrail Detection Research Problem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from ml_autoresearch.errors import TrainingError
from ml_autoresearch.training_adapters import ResearchProblemDatasets

_AUGMENTATION_SEED = 20260502


@dataclass(frozen=True)
class GVCCSTrainingAdapter:
    """Dataset/input-mode adapter for Ground-Camera Contrail Detection on GVCCS data."""

    def validate_data_root(self, data_config: Mapping[str, object]) -> Path:
        from ml_autoresearch.errors import GVCCSDataError

        data_root = _data_root(data_config)
        if not data_root.exists():
            raise GVCCSDataError(f"GVCCS data root does not exist: {data_root}")
        if not data_root.is_dir():
            raise GVCCSDataError(f"GVCCS data root is not a directory: {data_root}")
        return data_root.resolve()

    def dataset_metadata(self, data_config: Mapping[str, object]) -> dict[str, object]:
        data_root = _data_root(data_config).resolve()
        return {"id": "gvccs", "host_data_path": str(data_root), "container_data_path": "/data"}

    def build_datasets(
        self,
        *,
        data_config: Mapping[str, object],
        resolved_manifest_path: str | Path,
        max_samples: int | None = None,
    ) -> ResearchProblemDatasets:
        train_dataset, val_dataset, frame_selection_policy = self._build_split_datasets(
            data_config=data_config,
            resolved_manifest_path=resolved_manifest_path,
            max_samples=max_samples,
        )
        data_root = _data_root(data_config)
        return ResearchProblemDatasets(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            start_line=f"Starting GVCCS training from {data_root} with {len(train_dataset)} train and {len(val_dataset)} val samples.",
            success_line="GVCCS training completed.",
            failure_prefix="GVCCS training failed",
            data_policy_metadata={
                "frame_selection_policy": frame_selection_policy,
                "frame_selection_policy_effective": frame_selection_policy,
            },
        )

    def build_evaluation_dataset(
        self,
        *,
        data_config: Mapping[str, object],
        resolved_manifest_path: str | Path,
    ) -> object:
        _train_dataset, val_dataset, _frame_selection_policy = self._build_split_datasets(
            data_config=data_config,
            resolved_manifest_path=resolved_manifest_path,
            max_samples=None,
        )
        return val_dataset

    def apply_augmentation_policy(self, dataset: object, augmentation_policy: str) -> object:
        """Apply GVCCS-approved deterministic augmentation policy to a dataset."""

        if augmentation_policy == "none":
            return dataset
        if augmentation_policy in {"light_geometric", "light_photometric", "light_combined"}:
            return _AugmentedContrailDataset(dataset, augmentation_policy)
        raise TrainingError(f"unsupported augmentation policy: {augmentation_policy}")

    def primary_output_name(self, output_spec: Mapping[str, object]) -> str:
        return str(output_spec.get("form", "mask_logits"))

    def compute_primary_loss(self, loss_name: str, logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        from ml_autoresearch.problem_support.segmentation import bce_dice_loss

        if loss_name != "bce_dice":
            raise TrainingError(f"unsupported loss: {loss_name}")
        return bce_dice_loss(logits, target_mask)

    def compute_auxiliary_losses(
        self,
        outputs: dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        auxiliary_targets: list[dict[str, object]],
    ) -> dict[str, torch.Tensor]:
        from ml_autoresearch.problem_support.segmentation import weighted_bce_loss

        losses: dict[str, torch.Tensor] = {}
        for target in auxiliary_targets:
            name = str(target.get("name"))
            if target.get("loss") != "weighted_bce" or name not in {"line", "boundary"}:
                raise TrainingError(f"unsupported auxiliary target in resolved manifest: {target}")
            auxiliary_target = self.derive_auxiliary_target(name, target_mask)
            weight = float(target["weight"])
            losses[name] = weight * weighted_bce_loss(outputs[str(target["output"])], auxiliary_target)
        return losses

    def derive_auxiliary_target(self, name: str, target_mask: torch.Tensor) -> torch.Tensor:
        from ml_autoresearch.problem_support.segmentation import derive_boundary_target_v1, derive_line_target_v1

        if name == "line":
            return derive_line_target_v1(target_mask)
        if name == "boundary":
            return derive_boundary_target_v1(target_mask)
        raise TrainingError(f"unsupported auxiliary target: {name}")

    def compute_validation_metrics(self, logits: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
        import torch

        from ml_autoresearch.metrics import binary_segmentation_metrics

        metrics = binary_segmentation_metrics(torch.sigmoid(logits) >= 0.5, target_mask >= 0.5)
        return {
            "val/dice": metrics["dice"],
            "val/iou": metrics["iou"],
            "val/precision": metrics["precision"],
            "val/recall": metrics["recall"],
        }

    def selection_policy(self) -> tuple[str, str]:
        return "val/dice", "max"

    def _build_split_datasets(
        self,
        *,
        data_config: Mapping[str, object],
        resolved_manifest_path: str | Path,
        max_samples: int | None,
    ) -> tuple[object, object, str]:
        from ml_autoresearch.research_problem_packages.gvccs.datasets import (
            GVCCSDataset,
            GVCCSTemporalClipDataset,
            deterministic_train_val_split,
            discover_gvccs_samples,
            select_gvccs_frames,
        )

        data_root = _data_root(data_config)
        samples = discover_gvccs_samples(data_root, split="train", max_samples=max_samples)
        manifest = yaml.safe_load(Path(resolved_manifest_path).read_text())
        input_mode = str(manifest.get("input_mode", "single_frame_rgb"))
        data_policy = manifest.get("data", {}) or {}
        frame_selection_policy = str(
            data_policy.get("frame_selection_policy_effective")
            or data_policy.get("frame_selection_policy")
            or "all_target_frames"
        )
        if input_mode == "centered_temporal_rgb_clip":
            if frame_selection_policy != "temporal_eligible_center":
                raise TrainingError("centered_temporal_rgb_clip requires frame_selection_policy temporal_eligible_center")
            all_clips = GVCCSTemporalClipDataset(samples).clips
            split = deterministic_train_val_split(all_clips)  # type: ignore[arg-type]
            return GVCCSTemporalClipDataset(split.train), GVCCSTemporalClipDataset(split.val), frame_selection_policy  # type: ignore[arg-type]
        if input_mode == "single_frame_rgb":
            selected_samples = select_gvccs_frames(samples, frame_selection_policy)
            split = deterministic_train_val_split(selected_samples)
            return GVCCSDataset(split.train), GVCCSDataset(split.val), frame_selection_policy
        raise TrainingError(f"unsupported input mode for GVCCS training: {input_mode}")


class _AugmentedContrailDataset:
    """GVCCS deterministic augmentation wrapper for training samples."""

    def __init__(self, dataset: object, augmentation_policy: str) -> None:
        self.dataset = dataset
        self.augmentation_policy = augmentation_policy

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        image, mask = self.dataset[index]  # type: ignore[index]
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
    # remain deterministic under the Research Problem-owned policy.
    if index % 2 == 1:
        from ml_autoresearch.problem_support.imaging import horizontal_flip_image_mask

        return horizontal_flip_image_mask(image, mask)
    return image, mask


def _apply_light_photometric_augmentation(image: torch.Tensor, *, index: int) -> torch.Tensor:
    # Conservative GVCCS-specific brightness/contrast perturbation plus a tiny
    # deterministic sensor-noise term. The Contrail Mask is intentionally not
    # changed by photometric augmentation.
    contrast = 1.05 if index % 2 == 0 else 0.95
    brightness = 0.025 if index % 3 == 0 else -0.015
    from ml_autoresearch.problem_support.imaging import deterministic_photometric_perturbation

    return deterministic_photometric_perturbation(
        image,
        contrast=contrast,
        brightness=brightness,
        noise_seed=_AUGMENTATION_SEED + int(index),
    )


def build_spec(data_config: Mapping[str, object] | None = None):
    """Build the Ground-Camera Contrail Detection Research Problem Spec."""

    del data_config
    from ml_autoresearch.research_problems import DEFAULT_RESEARCH_PROBLEM_ID, ResearchProblemSpec

    return ResearchProblemSpec(
        id=DEFAULT_RESEARCH_PROBLEM_ID,
        version="v0",
        input_modes=("single_frame_rgb", "centered_temporal_rgb_clip"),
        input_specs={
            "single_frame_rgb": {"mode": "single_frame_rgb", "shape": [3, 128, 128]},
            "centered_temporal_rgb_clip": {
                "mode": "centered_temporal_rgb_clip",
                "shape": [9, 128, 128],
                "clip_length": 3,
                "frame_stride": 1,
                "layout": "channel_stacked_rgb",
                "target_frame": "center",
            },
        },
        output_forms=("mask_logits",),
        output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 128, 128]}},
        auxiliary_targets=("line", "boundary"),
        auxiliary_outputs={"line": "line_logits", "boundary": "boundary_logits"},
        auxiliary_output_shapes={"line": [1, 128, 128], "boundary": [1, 128, 128]},
        losses=("bce_dice",),
        auxiliary_losses=("weighted_bce",),
        optimizers=("adamw",),
        sampling_policies=("sequential", "deterministic_shuffle"),
        frame_selection_policies=("all_target_frames", "temporal_eligible_center"),
        input_mode_frame_selection_defaults={
            "single_frame_rgb": "all_target_frames",
            "centered_temporal_rgb_clip": "temporal_eligible_center",
        },
        augmentation_policies=("none", "light_geometric", "light_photometric", "light_combined"),
        primary_metric="val/dice",
        training_adapter=GVCCSTrainingAdapter(),
    )


def _data_root(data_config: Mapping[str, object]) -> Path:
    value = data_config.get("dataset_root") or data_config.get("data_root")
    if value is None:
        raise TrainingError("Research Problem data_config must include dataset_root")
    return Path(str(value))
