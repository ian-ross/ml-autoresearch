"""Manual Campaign Report and Campaign Pause Condition controls."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ml_autoresearch.research_ledger import (
    CAMPAIGN_PAUSE_CONDITIONS,
    CANONICAL_RESEARCH_LEDGER,
    record_research_event,
)

PauseCondition = Literal[
    "budget_exhausted",
    "repeated_failures",
    "repeated_resource_failures",
    "stalled_research_progress",
    "too_many_pending_capability_requests",
    "storage_risk",
    "scheduled_check_in",
]

PAUSE_CONDITIONS = CAMPAIGN_PAUSE_CONDITIONS


class CampaignControlError(ValueError):
    """Raised when a Campaign Report or pause control is invalid."""


def record_campaign_report_written(
    report_path: str | Path,
    *,
    ledger_path: str | Path = CANONICAL_RESEARCH_LEDGER,
) -> dict[str, object]:
    """Record that a Campaign Report artifact has been written."""

    normalized_report_path = _normalize_required_path(report_path, "report_path")
    event = record_research_event(
        "campaign_report_written",
        {"report_path": normalized_report_path},
        ledger_path=ledger_path,
    )
    return {"report_path": normalized_report_path, "ledger_event": event}


def record_campaign_pause(
    reason: str,
    *,
    report_path: str | Path | None = None,
    ledger_path: str | Path = CANONICAL_RESEARCH_LEDGER,
) -> dict[str, object]:
    """Record a campaign pause using the approved Campaign Pause Condition vocabulary."""

    if reason not in PAUSE_CONDITIONS:
        supported = ", ".join(PAUSE_CONDITIONS)
        raise CampaignControlError(f"unsupported Campaign Pause Condition '{reason}'; expected one of: {supported}")
    fields = {"reason": reason}
    normalized_report_path = None
    if report_path is not None:
        normalized_report_path = _normalize_required_path(report_path, "report_path")
        fields["report_path"] = normalized_report_path
    event = record_research_event("campaign_paused", fields, ledger_path=ledger_path)
    result: dict[str, object] = {"reason": reason, "ledger_event": event}
    if normalized_report_path is not None:
        result["report_path"] = normalized_report_path
    return result


def record_campaign_resume(
    reason: str,
    *,
    report_path: str | Path | None = None,
    ledger_path: str | Path = CANONICAL_RESEARCH_LEDGER,
) -> dict[str, object]:
    """Record that human review has cleared a Campaign Pause Condition."""

    normalized_reason = reason.strip() if isinstance(reason, str) else ""
    if not normalized_reason:
        raise CampaignControlError("resume reason must be non-empty")
    fields = {"reason": normalized_reason}
    normalized_report_path = None
    if report_path is not None:
        normalized_report_path = _normalize_required_path(report_path, "report_path")
        fields["report_path"] = normalized_report_path
    event = record_research_event("campaign_resumed", fields, ledger_path=ledger_path)
    result: dict[str, object] = {"reason": normalized_reason, "ledger_event": event}
    if normalized_report_path is not None:
        result["report_path"] = normalized_report_path
    return result


def _normalize_required_path(path: str | Path, field_name: str) -> str:
    value = str(path)
    if not value:
        raise CampaignControlError(f"{field_name} must be non-empty")
    return value
