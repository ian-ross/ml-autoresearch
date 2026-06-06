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

    def build_evaluation_dataset(
        self,
        *,
        data_config: Mapping[str, object],
        resolved_manifest_path: str | Path,
    ) -> object:
        """Build a validation dataset for Post-Run Evaluation."""
