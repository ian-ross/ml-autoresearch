"""Append-only Research Ledger recording for the Harness."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml_autoresearch.research_notes import ResearchFigureProvenance

CANONICAL_RESEARCH_LEDGER = "research-ledger.jsonl"


class ResearchLedgerError(ValueError):
    """Raised when a Research Ledger event cannot be validated or recorded."""


class _LedgerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str
    created_at: str


class ProposalCreated(_LedgerEvent):
    event_type: Literal["proposal_created"] = "proposal_created"
    proposal_id: str = Field(min_length=1)
    proposal_path: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)


RunFailureClassificationValue = Literal[
    "candidate_bug",
    "contract_violation",
    "resource_failure",
    "harness_failure",
    "bad_research_result",
    "unknown",
]


class RepairLineageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_proposal_id: str = Field(min_length=1)
    original_candidate_id: str = Field(min_length=1)
    motivating_run_id: str = Field(min_length=1)
    failure_classification: RunFailureClassificationValue
    preserves_original_hypothesis: Literal[True]
    preserves_comparison_target: Literal[True]


class CandidateCreated(_LedgerEvent):
    event_type: Literal["candidate_created"] = "candidate_created"
    candidate_id: str = Field(min_length=1)
    candidate_path: str = Field(min_length=1)
    proposal_id: str | None = Field(default=None, min_length=1)
    repair_lineage: RepairLineageRecord | None = None


class CandidateSubmitted(_LedgerEvent):
    event_type: Literal["candidate_submitted"] = "candidate_submitted"
    candidate_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


class RunStarted(_LedgerEvent):
    event_type: Literal["run_started"] = "run_started"
    run_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)


class RunCompleted(_LedgerEvent):
    event_type: Literal["run_completed"] = "run_completed"
    run_id: str = Field(min_length=1)
    metrics_path: str = Field(min_length=1)


class RunFailed(_LedgerEvent):
    event_type: Literal["run_failed"] = "run_failed"
    run_id: str = Field(min_length=1)
    error: str = Field(min_length=1)
    failure_classification: RunFailureClassificationValue = "unknown"


class ResearchNoteWritten(_LedgerEvent):
    event_type: Literal["research_note_written"] = "research_note_written"
    note_path: str = Field(min_length=1)
    run_id: str | None = Field(default=None, min_length=1)
    figure_provenance_path: str | None = Field(default=None, min_length=1)
    figure_provenance: list[ResearchFigureProvenance] | None = None


class CapabilityRequestCreated(_LedgerEvent):
    event_type: Literal["capability_request_created"] = "capability_request_created"
    request_id: str = Field(min_length=1)
    request_path: str = Field(min_length=1)


class EvaluationRequested(_LedgerEvent):
    event_type: Literal["evaluation_requested"] = "evaluation_requested"
    evaluation_request_id: str = Field(min_length=1)
    request_path: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    evaluation_mode: str = Field(min_length=1)


class EvaluationCompleted(_LedgerEvent):
    event_type: Literal["evaluation_completed"] = "evaluation_completed"
    evaluation_id: str = Field(min_length=1)
    evaluation_request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    evaluation_mode: str = Field(min_length=1)
    artifact_metadata_path: str = Field(min_length=1)


class CampaignReportWritten(_LedgerEvent):
    event_type: Literal["campaign_report_written"] = "campaign_report_written"
    report_path: str = Field(min_length=1)


CAMPAIGN_PAUSE_CONDITIONS: tuple[str, ...] = (
    "budget_exhausted",
    "repeated_failures",
    "repeated_resource_failures",
    "stalled_research_progress",
    "too_many_pending_capability_requests",
    "storage_risk",
    "scheduled_check_in",
)

CampaignPauseConditionValue = Literal[
    "budget_exhausted",
    "repeated_failures",
    "repeated_resource_failures",
    "stalled_research_progress",
    "too_many_pending_capability_requests",
    "storage_risk",
    "scheduled_check_in",
]


class CampaignPaused(_LedgerEvent):
    event_type: Literal["campaign_paused"] = "campaign_paused"
    reason: CampaignPauseConditionValue
    report_path: str | None = Field(default=None, min_length=1)


_EVENT_SCHEMAS: dict[str, type[_LedgerEvent]] = {
    "proposal_created": ProposalCreated,
    "candidate_created": CandidateCreated,
    "candidate_submitted": CandidateSubmitted,
    "run_started": RunStarted,
    "run_completed": RunCompleted,
    "run_failed": RunFailed,
    "research_note_written": ResearchNoteWritten,
    "capability_request_created": CapabilityRequestCreated,
    "evaluation_requested": EvaluationRequested,
    "evaluation_completed": EvaluationCompleted,
    "campaign_report_written": CampaignReportWritten,
    "campaign_paused": CampaignPaused,
}


def supported_research_event_types() -> tuple[str, ...]:
    """Return the Harness-supported Research Ledger event type vocabulary."""

    return tuple(_EVENT_SCHEMAS)


def record_research_event(
    event_type: str,
    fields: dict[str, Any],
    *,
    ledger_path: str | Path = CANONICAL_RESEARCH_LEDGER,
) -> dict[str, Any]:
    """Validate and atomically append one event to the append-only Research Ledger."""

    event = validate_research_event(event_type, fields)
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    _append_jsonl(Path(ledger_path), encoded)
    return event


def validate_research_event(event_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Validate a Research Ledger event and add Harness-owned metadata."""

    schema = _EVENT_SCHEMAS.get(event_type)
    if schema is None:
        supported = ", ".join(supported_research_event_types())
        raise ResearchLedgerError(f"unsupported event_type '{event_type}'; expected one of: {supported}")
    sanitized_fields = dict(fields)
    sanitized_fields.pop("created_at", None)
    payload = {"event_type": event_type, "created_at": _now_iso(), **sanitized_fields}
    try:
        event = schema.model_validate(payload)
    except ValidationError as exc:
        raise ResearchLedgerError(str(exc)) from exc
    return event.model_dump(exclude_none=True)


def _append_jsonl(path: Path, encoded_event: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, encoded_event.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
