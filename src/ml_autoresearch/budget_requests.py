"""Harness-owned Candidate budget request validation and resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

BUDGET_REQUEST_FILENAME = "budget_request.yaml"
BUDGET_REQUEST_SCHEMA_VERSION = "candidate_budget_request.v1"

BudgetRequestStatus = Literal["not_requested", "approved", "denied"]
BudgetClass = Literal["exploratory", "late_improving"]


class CandidateBudgetRequest(BaseModel):
    """Machine-readable request for a Harness-approved non-default budget class."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["candidate_budget_request.v1"]
    budget_class: Literal["late_improving"]
    requested_max_epochs: int = Field(ge=1, le=100)
    parent_run_ids: list[str] = Field(min_length=1)
    justification: str = Field(min_length=1)
    fallback_if_denied: Literal["run_exploratory_budget"] = "run_exploratory_budget"


@dataclass(frozen=True)
class CandidateBudgetPolicy:
    """Harness-owned two-tier Candidate training budget policy."""

    exploratory_max_epochs: int
    exploratory_training_wall_clock_timeout_seconds: int | None = None
    late_improving_max_epochs: int | None = None
    late_improving_training_wall_clock_timeout_seconds: int | None = None


@dataclass(frozen=True)
class CandidateBudgetDecision:
    """Resolved effective budget class for one Candidate Run."""

    status: BudgetRequestStatus
    requested_class: str
    effective_class: BudgetClass
    effective_max_epochs: int
    effective_training_wall_clock_timeout_seconds: int | None
    requested_max_epochs: int | None = None
    parent_run_ids: tuple[str, ...] = ()
    reason: str | None = None
    denial_reason: str | None = None

    def with_source_manifest_max_epochs(self, source_max_epochs: int) -> "CandidateBudgetDecision":
        """Return a decision whose exploratory fallback matches the source manifest's bounded epochs."""

        if self.status == "approved":
            return self
        return replace(self, effective_max_epochs=source_max_epochs)

    def metadata(self) -> dict[str, object]:
        """Return auditable Run metadata fields for this decision."""

        fields: dict[str, object] = {
            "budget_class_requested": self.requested_class,
            "budget_class_effective": self.effective_class,
            "effective_max_epochs": self.effective_max_epochs,
            "effective_training_wall_clock_timeout_seconds": self.effective_training_wall_clock_timeout_seconds,
            "budget_request_status": self.status,
            "budget_request_parent_run_ids": list(self.parent_run_ids),
        }
        if self.requested_max_epochs is not None:
            fields["budget_request_requested_max_epochs"] = self.requested_max_epochs
        if self.reason is not None:
            fields["budget_request_reason"] = self.reason
        if self.denial_reason is not None:
            fields["budget_request_denial_reason"] = self.denial_reason
        return fields

    def resolved_manifest_payload(self) -> dict[str, object]:
        """Return the resolved-manifest budget audit payload."""

        return self.metadata()


def decide_candidate_budget(
    candidate_dir: str | Path,
    policy: CandidateBudgetPolicy,
    *,
    runs_root: str | Path | None = None,
) -> CandidateBudgetDecision:
    """Validate and resolve an optional ``budget_request.yaml`` for one Candidate.

    Malformed or denied requests fall back to the exploratory budget. Candidate
    manifests are still validated separately against the exploratory source
    contract, so the request file cannot by itself bypass Harness policy.
    """

    request_path = Path(candidate_dir) / BUDGET_REQUEST_FILENAME
    if not request_path.is_file():
        return CandidateBudgetDecision(
            status="not_requested",
            requested_class="exploratory",
            effective_class="exploratory",
            effective_max_epochs=policy.exploratory_max_epochs,
            effective_training_wall_clock_timeout_seconds=policy.exploratory_training_wall_clock_timeout_seconds,
            reason="exploratory_default",
        )

    try:
        request = validate_candidate_budget_request_file(request_path)
    except ValueError as exc:
        return CandidateBudgetDecision(
            status="denied",
            requested_class="unknown",
            effective_class="exploratory",
            effective_max_epochs=policy.exploratory_max_epochs,
            effective_training_wall_clock_timeout_seconds=policy.exploratory_training_wall_clock_timeout_seconds,
            denial_reason=f"malformed_request: {exc}",
        )

    if policy.late_improving_max_epochs is None:
        return _deny_request(request, policy, "late_improving_budget_not_configured")
    if request.requested_max_epochs > policy.late_improving_max_epochs:
        return _deny_request(request, policy, "requested_max_epochs_exceeds_late_improving_limit")
    if request.requested_max_epochs <= policy.exploratory_max_epochs:
        return _deny_request(request, policy, "requested_max_epochs_not_above_exploratory_ceiling")
    if runs_root is not None:
        evidence_denial = _parent_run_evidence_denial(request, Path(runs_root))
        if evidence_denial is not None:
            return _deny_request(request, policy, evidence_denial)

    return CandidateBudgetDecision(
        status="approved",
        requested_class=request.budget_class,
        effective_class="late_improving",
        requested_max_epochs=request.requested_max_epochs,
        effective_max_epochs=request.requested_max_epochs,
        effective_training_wall_clock_timeout_seconds=policy.late_improving_training_wall_clock_timeout_seconds,
        parent_run_ids=tuple(request.parent_run_ids),
        reason="late_improving_parent_completed",
    )


def _parent_run_evidence_denial(request: CandidateBudgetRequest, runs_root: Path) -> str | None:
    for parent_run_id in request.parent_run_ids:
        metadata_path = runs_root / parent_run_id / "run_metadata.json"
        if not metadata_path.is_file():
            return "parent_run_metadata_missing"
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            return "parent_run_metadata_unreadable"
        if not isinstance(metadata, dict) or metadata.get("status") != "completed":
            return "parent_run_not_completed"
        if metadata.get("failure_classification") not in {None, ""}:
            return "parent_run_has_failure_classification"
    return None


def validate_candidate_budget_request_file(path: str | Path) -> CandidateBudgetRequest:
    """Parse and validate a Candidate budget request file."""

    request_path = Path(path)
    try:
        loaded = yaml.safe_load(request_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid {BUDGET_REQUEST_FILENAME}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read {BUDGET_REQUEST_FILENAME}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{BUDGET_REQUEST_FILENAME} must contain a mapping")
    try:
        return CandidateBudgetRequest.model_validate(loaded)
    except ValidationError as exc:
        details = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            details.append(f"{loc}: {error['msg']}")
        raise ValueError("invalid budget_request.yaml: " + "; ".join(details)) from exc


def apply_budget_decision_to_resolved_manifest(
    resolved_manifest: dict[str, object],
    decision: CandidateBudgetDecision,
) -> dict[str, object]:
    """Apply the Harness-owned effective epoch budget to a resolved manifest payload."""

    updated = dict(resolved_manifest)
    training = dict(updated.get("training") or {})
    training["max_epochs"] = decision.effective_max_epochs
    updated["training"] = training
    updated["budget"] = decision.resolved_manifest_payload()
    return updated


def _deny_request(
    request: CandidateBudgetRequest,
    policy: CandidateBudgetPolicy,
    reason: str,
) -> CandidateBudgetDecision:
    return CandidateBudgetDecision(
        status="denied",
        requested_class=request.budget_class,
        effective_class="exploratory",
        requested_max_epochs=request.requested_max_epochs,
        effective_max_epochs=policy.exploratory_max_epochs,
        effective_training_wall_clock_timeout_seconds=policy.exploratory_training_wall_clock_timeout_seconds,
        parent_run_ids=tuple(request.parent_run_ids),
        denial_reason=reason,
    )
