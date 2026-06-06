"""Trusted Research Problem training adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import yaml

from ml_autoresearch.errors import TrainingError


@dataclass(frozen=True)
class ResearchProblemDatasets:
    """Datasets and metadata supplied by a trusted Research Problem adapter."""

    train_dataset: object
    validation_dataset: object
    start_line: str
    success_line: str
    failure_prefix: str
    data_policy_metadata: dict[str, object]


class ResearchProblemTrainingAdapter(Protocol):
    """Trusted adapter used by the generic Harness training loop."""

    def dataset_metadata(self, data_config: Mapping[str, object]) -> dict[str, object]:
        """Return JSON-serialisable dataset provenance for Run metadata."""

    def build_datasets(
        self,
        *,
        data_config: Mapping[str, object],
        resolved_manifest_path: str | Path,
        max_samples: int | None = None,
    ) -> ResearchProblemDatasets:
        """Build training/validation datasets for a Resolved Manifest."""


@dataclass(frozen=True)
class GVCCSTrainingAdapter:
    """Training adapter for Ground-Camera Contrail Detection on GVCCS data."""

    def dataset_metadata(self, data_config: Mapping[str, object]) -> dict[str, object]:
        data_root = _data_root(data_config)
        return {"id": "gvccs", "host_data_path": str(data_root), "container_data_path": "/data"}

    def build_datasets(
        self,
        *,
        data_config: Mapping[str, object],
        resolved_manifest_path: str | Path,
        max_samples: int | None = None,
    ) -> ResearchProblemDatasets:
        from ml_autoresearch.gvccs import (
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
            train_dataset = GVCCSTemporalClipDataset(split.train)  # type: ignore[arg-type]
            val_dataset = GVCCSTemporalClipDataset(split.val)  # type: ignore[arg-type]
        elif input_mode == "single_frame_rgb":
            selected_samples = select_gvccs_frames(samples, frame_selection_policy)
            split = deterministic_train_val_split(selected_samples)
            train_dataset = GVCCSDataset(split.train)
            val_dataset = GVCCSDataset(split.val)
        else:
            raise TrainingError(f"unsupported input mode for GVCCS training: {input_mode}")

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


def _data_root(data_config: Mapping[str, object]) -> Path:
    value = data_config.get("dataset_root") or data_config.get("data_root")
    if value is None:
        raise TrainingError("Research Problem data_config must include dataset_root")
    return Path(str(value))
