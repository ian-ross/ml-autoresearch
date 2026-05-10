"""Validated Capability Request workflow for Harness-owned changes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, record_research_event


class CapabilityRequestError(ValueError):
    """Raised when a Capability Request cannot be validated or recorded."""


CapabilityType = Literal["contract_surface", "approved_resource", "operational_policy"]
Priority = Literal["low", "medium", "high", "urgent"]
CandidateAuthorityRequested = Literal["none", "read_only_harness_metadata", "other"]


class CapabilityRequest(BaseModel):
    """Human-reviewable request to expand Harness-owned capability surface."""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, min_length=1)
    capability_type: CapabilityType
    blocked_hypothesis: str = Field(min_length=1)
    current_contract_insufficiency: str = Field(min_length=1)
    expected_research_value: str = Field(min_length=1)
    safety_reproducibility_risks: str = Field(min_length=1)
    minimal_harness_change: str = Field(min_length=1)
    candidate_authority_requested: CandidateAuthorityRequested
    example_follow_up_experiments: list[str] = Field(min_length=1)
    priority: Priority

    @field_validator("example_follow_up_experiments")
    @classmethod
    def _examples_must_be_non_empty(cls, examples: list[str]) -> list[str]:
        if any(not example.strip() for example in examples):
            raise ValueError("example_follow_up_experiments entries must be non-empty")
        return examples


def validate_capability_request_file(request_path: str | Path) -> CapabilityRequest:
    """Load and validate one YAML Capability Request file."""

    path = Path(request_path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise CapabilityRequestError(f"cannot read Capability Request {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CapabilityRequestError(f"invalid Capability Request YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CapabilityRequestError("Capability Request must be a YAML mapping")
    try:
        request = CapabilityRequest.model_validate(raw)
    except ValidationError as exc:
        raise CapabilityRequestError(str(exc)) from exc
    if request.request_id is None:
        request = request.model_copy(update={"request_id": path.stem})
    return request


def create_capability_request(
    request_path: str | Path,
    *,
    ledger_path: str | Path = CANONICAL_RESEARCH_LEDGER,
) -> dict[str, Any]:
    """Validate a Capability Request and record a Research Ledger creation event.

    Recording this event does not approve or apply the requested Harness change;
    it only creates an auditable request for later human-supervised review.
    """

    path = Path(request_path)
    request = validate_capability_request_file(path)
    event = record_research_event(
        "capability_request_created",
        {"request_id": request.request_id, "request_path": str(path)},
        ledger_path=ledger_path,
    )
    return {"request": request.model_dump(), "ledger_event": event}
