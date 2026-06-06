"""Trusted Research Problem training adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


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

    def validate_data_root(self, data_config: Mapping[str, object]) -> Path:
        """Validate and resolve the configured Research Problem data root."""

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

    def apply_augmentation_policy(self, dataset: object, augmentation_policy: str) -> object:
        """Apply a Research Problem-owned augmentation policy to a dataset."""

    def primary_output_name(self, output_spec: Mapping[str, object]) -> str:
        """Return the primary model output key for training/evaluation."""

    def compute_primary_loss(self, loss_name: str, logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Compute a Research Problem-approved primary training loss."""

    def compute_auxiliary_losses(
        self,
        outputs: dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        auxiliary_targets: list[dict[str, object]],
    ) -> dict[str, torch.Tensor]:
        """Derive auxiliary targets and compute weighted auxiliary losses."""

    def compute_validation_metrics(self, logits: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
        """Return validation metrics for primary model output selection/reporting."""

    def selection_policy(self) -> tuple[str, str]:
        """Return ``(metric_name, mode)`` used for best-epoch selection."""

    def build_evaluation_dataset(
        self,
        *,
        data_config: Mapping[str, object],
        resolved_manifest_path: str | Path,
    ) -> object:
        """Build a validation dataset for Post-Run Evaluation."""
