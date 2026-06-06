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
class GVCCSEvaluationAdapter:
    """Evaluation adapter for Ground-Camera Contrail Detection on GVCCS data."""

    supported_evaluation_modes = ("whole_validation_failure_analysis",)

    def run_evaluation_mode(
        self,
        *,
        mode: str,
        run_dir: Path,
        data_root: Path,
        model_artifact_path: Path,
        threshold: float,
        evaluation_dir: Path,
        max_artifact_samples: int,
    ):
        if mode != "whole_validation_failure_analysis":
            from ml_autoresearch.evaluations import EvaluationError

            raise EvaluationError(f"unsupported evaluation mode for Ground-Camera Contrail Detection: {mode}")
        from ml_autoresearch.evaluations import evaluate_binary_segmentation_validation_split

        return evaluate_binary_segmentation_validation_split(
            adapter=GVCCSTrainingAdapter(),
            run_dir=run_dir,
            data_root=data_root,
            model_artifact_path=model_artifact_path,
            threshold=threshold,
            evaluation_dir=evaluation_dir,
            max_artifact_samples=max_artifact_samples,
        )


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

    def select_prediction_samples(self, dataset: object, *, policy: str, max_samples: int) -> list[dict[str, object]]:
        """Select bounded qualitative prediction samples for GVCCS validation datasets."""

        samples = getattr(dataset, "samples", None)
        if not isinstance(samples, list):
            from ml_autoresearch.artifacts import select_prediction_sample_indices

            return select_prediction_sample_indices(list(range(len(dataset))), policy="first_n", max_samples=max_samples)  # type: ignore[arg-type]
        if policy == "first_n":
            from ml_autoresearch.artifacts import select_prediction_sample_indices

            return select_prediction_sample_indices(samples, policy="first_n", max_samples=max_samples)
        if policy != "adjacent_and_scattered":
            raise TrainingError(f"unsupported prediction sample policy: {policy}")
        return _select_adjacent_and_scattered_prediction_samples(samples, max_samples=max_samples)

    def display_prediction_sample_input(self, inputs: torch.Tensor) -> torch.Tensor:
        """Choose the GVCCS RGB Target Frame view for qualitative prediction figures."""

        if inputs.ndim == 3 and inputs.shape[0] == 3:
            return inputs
        if inputs.ndim == 3 and inputs.shape[0] == 9:
            return inputs[3:6]
        raise TrainingError(f"cannot render prediction sample input with shape {tuple(inputs.shape)} as RGB")

    def write_prediction_sample_images(
        self,
        samples_dir: Path,
        paths: dict[str, str],
        image: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> None:
        """Write GVCCS Contrail Mask qualitative figure images."""

        from ml_autoresearch.problem_support.imaging import (
            save_mask_tensor,
            save_overlay,
            save_probability_heatmap,
            save_rgb_tensor,
        )

        save_rgb_tensor(samples_dir / paths["input"], image)
        save_mask_tensor(samples_dir / paths["ground_truth"], target)
        save_mask_tensor(samples_dir / paths["prediction"], prediction)
        save_overlay(samples_dir / paths["overlay"], image, target, prediction)
        save_probability_heatmap(samples_dir / paths["probability_heatmap"], probabilities)

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


def _select_adjacent_and_scattered_prediction_samples(samples: list[object], *, max_samples: int) -> list[dict[str, object]]:
    from ml_autoresearch.problem_support.frame_sequences import infer_timestamped_frame_sequences

    index_by_identity = {id(sample): index for index, sample in enumerate(samples)}
    sequences = infer_timestamped_frame_sequences(samples, filename_for_item=lambda sample: getattr(sample, "image_path", ""))
    eligible_sequences = [sequence for sequence in sequences if len(sequence) >= 2 and any(_is_positive_sample(sample) for sample in sequence)]
    selections: list[dict[str, object]] = []

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
            frame_sequence_id = Path(getattr(sequence[0], "image_path")).stem
            window_id = f"{frame_sequence_id}/window_{window_number:03d}"
            for offset, sample in enumerate(window):
                if len(selections) >= adjacent_budget:
                    break
                selections.append(
                    _prediction_selection(
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
        if any(_is_positive_sample(sample) for sample in window):
            return window
    return sequence[:window_length]


def _scattered_singletons(samples: list[object], budget: int, *, already_selected: set[int]) -> list[dict[str, object]]:
    available_positive = [index for index, sample in enumerate(samples) if index not in already_selected and _is_positive_sample(sample)]
    available_negative = [index for index, sample in enumerate(samples) if index not in already_selected and not _is_positive_sample(sample)]
    negative_count = 1 if budget >= 2 and available_negative else 0
    positive_count = min(len(available_positive), budget - negative_count)
    indices = _spread_values(available_positive, positive_count)
    indices.extend(_spread_values(available_negative, min(negative_count, budget - len(indices))))
    if len(indices) < budget:
        remaining = [index for index in range(len(samples)) if index not in already_selected and index not in indices]
        indices.extend(_spread_values(remaining, budget - len(indices)))
    return [_prediction_selection(index, "scattered_singleton") for index in indices[:budget]]


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


def _prediction_selection(dataset_index: int, selection_kind: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"dataset_index": dataset_index, "selection_kind": selection_kind}
    payload.update(extra)
    return payload


def _is_positive_sample(sample: object) -> bool:
    return bool(getattr(sample, "segmentations", ()))


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
        evaluation_adapter=GVCCSEvaluationAdapter(),
    )


def _data_root(data_config: Mapping[str, object]) -> Path:
    value = data_config.get("dataset_root") or data_config.get("data_root")
    if value is None:
        raise TrainingError("Research Problem data_config must include dataset_root")
    return Path(str(value))
