"""Harness-owned Agent Handoff Ingestion."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml_autoresearch.campaign_controls import PAUSE_CONDITIONS
from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.capability_requests import CapabilityRequestError, validate_capability_request_file
from ml_autoresearch.evaluation_requests import EvaluationRequestError, validate_evaluation_request_file
from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, record_research_event
from ml_autoresearch.research_notes import ResearchNoteFigureError, validate_research_note_figure_provenance
from ml_autoresearch.submissions import (
    RELATIVE_CANDIDATE_PATH,
    REQUESTED_ACTION,
    SUBMISSION_SCHEMA_VERSION,
    SUBMISSION_TYPE,
)

INGESTION_MARKER = "INGESTED.json"

PRIMARY_HANDOFF_TYPES = (
    "candidate_submission",
    "evaluation_request",
    "capability_request",
    "research_note",
    "campaign_report",
)


class AgentHandoffIngestionError(ValueError):
    """Raised when a Harness handoff cannot be ingested safely."""


class CandidateSubmissionMetadata(BaseModel):
    """Validated Agent Workspace Candidate Submission metadata."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    submission_type: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    candidate_path: str = Field(min_length=1)
    requested_action: str = Field(min_length=1)


def collect_agent_handoff(project_root: str | Path = Path(".")) -> dict[str, object]:
    """Collect and ingest exactly one primary Agent Workspace handoff for an Autonomy Step."""

    root = Path(project_root).resolve()
    artifacts = _discover_uningested_primary_handoffs(root)
    present = [handoff_type for handoff_type in PRIMARY_HANDOFF_TYPES if artifacts[handoff_type]]

    if not present:
        return {
            "status": "no_handoff",
            "next_action": "stop_for_human",
            "executed_next_action": False,
            "ledger_events": [],
        }

    if len(present) > 1:
        return _ingestion_failed(
            "multiple primary handoff categories produced in one Autonomy Step: " + ", ".join(present),
            present,
        )

    handoff_type = present[0]
    if len(artifacts[handoff_type]) > 1:
        return _ingestion_failed(
            f"multiple un-ingested {handoff_type} artifacts produced in one Autonomy Step: "
            + ", ".join(_relative_posix(path, root) for path in artifacts[handoff_type]),
            [handoff_type],
        )

    ingest = {
        "candidate_submission": ingest_candidate_submission,
        "evaluation_request": ingest_evaluation_request,
        "capability_request": ingest_capability_request,
        "research_note": ingest_research_note,
        "campaign_report": ingest_campaign_report,
    }[handoff_type]
    try:
        return ingest(root)
    except AgentHandoffIngestionError as exc:
        return _ingestion_failed(str(exc), [handoff_type])


def ingest_candidate_submission(project_root: str | Path = Path(".")) -> dict[str, object]:
    """Ingest exactly one un-ingested Candidate Submission without executing it."""

    root = Path(project_root).resolve()
    submissions_root = root / "agent-work" / "submissions"
    source_dir = _discover_one_uningested_submission(submissions_root)
    metadata = _load_submission_metadata(source_dir)
    candidate_id = metadata.candidate_id
    source_candidate = _resolve_source_candidate(source_dir, metadata)
    canonical_candidate = root / "candidates" / candidate_id

    if canonical_candidate.exists():
        raise AgentHandoffIngestionError(f"canonical Candidate Experiment already exists: {canonical_candidate}")

    try:
        manifest = validate_candidate_directory(source_candidate, require_proposal=True, require_readme=True)
    except CandidateValidationError as exc:
        raise AgentHandoffIngestionError(str(exc)) from exc
    if manifest.name != candidate_id:
        raise AgentHandoffIngestionError(f"manifest name must match submission candidate_id '{candidate_id}' (got '{manifest.name}')")

    canonical_candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_candidate, canonical_candidate)
    canonical_relative = _relative_posix(canonical_candidate, root)
    source_relative = _relative_posix(source_dir, root)

    _append_candidate_to_experiment_index(root / "EXPERIMENT_INDEX.md", candidate_id, manifest.description)
    event = record_research_event(
        "agent_handoff_ingested",
        {
            "handoff_type": "candidate_submission",
            "artifact_id": candidate_id,
            "source_path": source_relative,
            "canonical_path": canonical_relative,
            "candidate_id": candidate_id,
        },
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    _write_ingestion_marker_last(source_dir, candidate_id, canonical_relative)

    return {
        "status": "ingested",
        "handoff_type": "candidate_submission",
        "candidate_id": candidate_id,
        "source_path": source_relative,
        "canonical_path": canonical_relative,
        "ledger_event": event,
        "next_action": "run_candidate",
        "executed_next_action": False,
    }


def ingest_research_note(project_root: str | Path = Path(".")) -> dict[str, object]:
    """Ingest exactly one un-ingested Agent Workspace Research Note."""

    root = Path(project_root).resolve()
    notes_root = root / "agent-work" / "research-notes"
    source_note = _discover_one_uningested_research_note(notes_root)
    note_text = _validate_research_note(source_note)
    canonical_note = root / "research-notes" / source_note.name

    if canonical_note.exists():
        raise AgentHandoffIngestionError(f"canonical Research Note already exists: {canonical_note}")

    canonical_note.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_note, canonical_note)
    canonical_relative = _relative_posix(canonical_note, root)
    source_relative = _relative_posix(source_note, root)
    title = _research_note_title(note_text, source_note)

    _append_research_note_to_experiment_index(root / "EXPERIMENT_INDEX.md", canonical_relative, title)
    handoff_event = record_research_event(
        "agent_handoff_ingested",
        {
            "handoff_type": "research_note",
            "artifact_id": source_note.name,
            "source_path": source_relative,
            "canonical_path": canonical_relative,
            "note_path": canonical_relative,
        },
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    note_event = record_research_event(
        "research_note_written",
        {"note_path": canonical_relative},
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    _write_research_note_ingestion_marker_last(source_note, canonical_relative)

    return {
        "status": "ingested",
        "handoff_type": "research_note",
        "note_path": canonical_relative,
        "source_path": source_relative,
        "canonical_path": canonical_relative,
        "ledger_events": [handoff_event, note_event],
        "next_action": "continue_autonomy",
        "executed_next_action": False,
    }


def ingest_capability_request(project_root: str | Path = Path(".")) -> dict[str, object]:
    """Ingest exactly one un-ingested Agent Workspace Capability Request."""

    root = Path(project_root).resolve()
    source = _discover_one_uningested_file(root / "agent-work" / "capability-requests", "Capability Request", {".yaml", ".yml"})
    try:
        request = validate_capability_request_file(source)
    except CapabilityRequestError as exc:
        raise AgentHandoffIngestionError(str(exc)) from exc
    assert request.request_id is not None
    canonical = root / "capability-requests" / source.name
    if canonical.exists():
        raise AgentHandoffIngestionError(f"canonical Capability Request already exists: {canonical}")

    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, canonical)
    canonical_relative = _relative_posix(canonical, root)
    source_relative = _relative_posix(source, root)
    handoff_event = record_research_event(
        "agent_handoff_ingested",
        {
            "handoff_type": "capability_request",
            "artifact_id": request.request_id,
            "source_path": source_relative,
            "canonical_path": canonical_relative,
            "request_id": request.request_id,
        },
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    request_event = record_research_event(
        "capability_request_created",
        {"request_id": request.request_id, "request_path": canonical_relative},
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    _write_file_ingestion_marker_last(source, "capability_request", request.request_id, canonical_relative)
    return {
        "status": "ingested",
        "handoff_type": "capability_request",
        "request_id": request.request_id,
        "source_path": source_relative,
        "canonical_path": canonical_relative,
        "ledger_events": [handoff_event, request_event],
        "next_action": "stop_for_human",
        "executed_next_action": False,
    }


def ingest_evaluation_request(project_root: str | Path = Path(".")) -> dict[str, object]:
    """Ingest exactly one un-ingested Agent Workspace Evaluation Request without executing it."""

    root = Path(project_root).resolve()
    source = _discover_one_uningested_file(root / "agent-work" / "evaluation-requests", "Evaluation Request", {".yaml", ".yml"})
    try:
        request = validate_evaluation_request_file(source)
    except EvaluationRequestError as exc:
        raise AgentHandoffIngestionError(str(exc)) from exc
    assert request.request_id is not None
    canonical = root / "evaluation-requests" / source.name
    if canonical.exists():
        raise AgentHandoffIngestionError(f"canonical Evaluation Request already exists: {canonical}")

    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, canonical)
    canonical_relative = _relative_posix(canonical, root)
    source_relative = _relative_posix(source, root)
    handoff_event = record_research_event(
        "agent_handoff_ingested",
        {
            "handoff_type": "evaluation_request",
            "artifact_id": request.request_id,
            "source_path": source_relative,
            "canonical_path": canonical_relative,
            "request_id": request.request_id,
            "run_id": request.target_run_id,
        },
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    _write_file_ingestion_marker_last(source, "evaluation_request", request.request_id, canonical_relative)
    return {
        "status": "ingested",
        "handoff_type": "evaluation_request",
        "request_id": request.request_id,
        "run_id": request.target_run_id,
        "source_path": source_relative,
        "canonical_path": canonical_relative,
        "ledger_events": [handoff_event],
        "next_action": "run_post_run_evaluation",
        "executed_next_action": False,
    }


def ingest_campaign_report(project_root: str | Path = Path(".")) -> dict[str, object]:
    """Ingest exactly one un-ingested Agent Workspace Campaign Report."""

    root = Path(project_root).resolve()
    source = _discover_one_uningested_file(root / "agent-work" / "campaign-reports", "Campaign Report", {".md"})
    text = _validate_campaign_report(source)
    pause_condition = _campaign_report_pause_condition(text)
    canonical = root / "campaign-reports" / source.name
    if canonical.exists():
        raise AgentHandoffIngestionError(f"canonical Campaign Report already exists: {canonical}")

    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, canonical)
    canonical_relative = _relative_posix(canonical, root)
    source_relative = _relative_posix(source, root)
    handoff_event = record_research_event(
        "agent_handoff_ingested",
        {
            "handoff_type": "campaign_report",
            "artifact_id": source.name,
            "source_path": source_relative,
            "canonical_path": canonical_relative,
            "report_path": canonical_relative,
        },
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    report_event = record_research_event(
        "campaign_report_written",
        {"report_path": canonical_relative},
        ledger_path=root / CANONICAL_RESEARCH_LEDGER,
    )
    _write_file_ingestion_marker_last(source, "campaign_report", source.name, canonical_relative)
    return {
        "status": "ingested",
        "handoff_type": "campaign_report",
        "report_path": canonical_relative,
        "source_path": source_relative,
        "canonical_path": canonical_relative,
        "ledger_events": [handoff_event, report_event],
        "next_action": "pause_campaign" if pause_condition is not None else "stop_for_human",
        "executed_next_action": False,
    }


def _ingestion_failed(reason: str, handoff_types: list[str]) -> dict[str, object]:
    return {
        "status": "ingestion_failed",
        "reason": reason,
        "handoff_types": handoff_types,
        "next_action": "stop_for_human",
        "executed_next_action": False,
        "ledger_events": [],
    }


def _discover_uningested_primary_handoffs(root: Path) -> dict[str, list[Path]]:
    return {
        "candidate_submission": _list_uningested_submissions(root / "agent-work" / "submissions"),
        "evaluation_request": _list_uningested_files(
            root / "agent-work" / "evaluation-requests", {".yaml", ".yml"}, "evaluation_request"
        ),
        "capability_request": _list_uningested_files(
            root / "agent-work" / "capability-requests", {".yaml", ".yml"}, "capability_request"
        ),
        "research_note": _list_uningested_research_notes(root / "agent-work" / "research-notes"),
        "campaign_report": _list_uningested_files(root / "agent-work" / "campaign-reports", {".md"}, "campaign_report"),
    }


def _list_uningested_submissions(submissions_root: Path) -> list[Path]:
    if not submissions_root.is_dir():
        return []
    return sorted(
        path
        for path in submissions_root.iterdir()
        if path.is_dir() and not _has_valid_ingestion_marker(path / INGESTION_MARKER, "candidate_submission")
    )


def _list_uningested_files(directory: Path, suffixes: set[str], handoff_type: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in suffixes and not _has_valid_ingestion_marker(_file_marker_path(path), handoff_type)
    )


def _list_uningested_research_notes(notes_root: Path) -> list[Path]:
    if not notes_root.is_dir():
        return []
    return sorted(
        path
        for path in notes_root.glob("*.md")
        if path.is_file() and not _has_valid_ingestion_marker(_research_note_marker_path(path), "research_note")
    )


def _discover_one_uningested_submission(submissions_root: Path) -> Path:
    if not submissions_root.is_dir():
        raise AgentHandoffIngestionError(f"Candidate Submission Queue does not exist: {submissions_root}")
    submissions = _list_uningested_submissions(submissions_root)
    if len(submissions) != 1:
        raise AgentHandoffIngestionError(
            f"expected exactly one un-ingested Candidate Submission in {submissions_root}, found {len(submissions)}"
        )
    return submissions[0]


def _discover_one_uningested_file(directory: Path, handoff_name: str, suffixes: set[str]) -> Path:
    if not directory.is_dir():
        raise AgentHandoffIngestionError(f"Agent Workspace {handoff_name} directory does not exist: {directory}")
    handoff_type = handoff_name.lower().replace(" ", "_")
    files = _list_uningested_files(directory, suffixes, handoff_type)
    if len(files) != 1:
        raise AgentHandoffIngestionError(f"expected exactly one un-ingested {handoff_name} in {directory}, found {len(files)}")
    return files[0]


def _discover_one_uningested_research_note(notes_root: Path) -> Path:
    if not notes_root.is_dir():
        raise AgentHandoffIngestionError(f"Agent Workspace Research Notes directory does not exist: {notes_root}")
    notes = _list_uningested_research_notes(notes_root)
    if len(notes) != 1:
        raise AgentHandoffIngestionError(
            f"expected exactly one un-ingested Research Note in {notes_root}, found {len(notes)}"
        )
    return notes[0]


def _validate_campaign_report(report_path: Path) -> str:
    try:
        text = report_path.read_text()
    except OSError as exc:
        raise AgentHandoffIngestionError(f"cannot read Campaign Report: {exc}") from exc
    if not text.strip():
        raise AgentHandoffIngestionError(f"Campaign Report is empty: {report_path}")
    required_headings = [
        "## Summary",
        "## Current best Result",
        "## Recent Runs",
        "## Failures",
        "## Pending Capability Requests",
        "## Budget use",
        "## Next hypothesis",
        "## Pause recommendation",
    ]
    missing = [heading for heading in required_headings if heading not in text]
    if missing:
        raise AgentHandoffIngestionError(f"Campaign Report missing required headings: {', '.join(missing)}")
    _campaign_report_pause_condition(text)
    return text


def _campaign_report_pause_condition(report_text: str) -> str | None:
    for line in report_text.splitlines():
        normalized = line.strip()
        prefix = "- Pause condition:"
        if normalized.startswith(prefix):
            value = normalized[len(prefix) :].strip().strip("`")
            if value in {"", "none"}:
                return None
            if value not in PAUSE_CONDITIONS:
                supported = ", ".join(["none", *PAUSE_CONDITIONS])
                raise AgentHandoffIngestionError(
                    f"Campaign Report Pause condition must be one of: {supported} (got {value!r})"
                )
            return value
    raise AgentHandoffIngestionError("Campaign Report missing Pause condition line")


def _validate_research_note(note_path: Path) -> str:
    try:
        text = note_path.read_text()
    except OSError as exc:
        raise AgentHandoffIngestionError(f"cannot read Research Note: {exc}") from exc
    if not text.strip():
        raise AgentHandoffIngestionError(f"Research Note is empty: {note_path}")
    try:
        validate_research_note_figure_provenance(text)
    except ResearchNoteFigureError as exc:
        raise AgentHandoffIngestionError(f"Research Figure provenance is malformed: {exc}") from exc
    return text


def _load_submission_metadata(source_dir: Path) -> CandidateSubmissionMetadata:
    metadata_path = source_dir / "submission.json"
    try:
        raw: Any = json.loads(metadata_path.read_text())
    except OSError as exc:
        raise AgentHandoffIngestionError(f"cannot read submission.json: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AgentHandoffIngestionError(f"invalid submission.json: {exc}") from exc
    try:
        metadata = CandidateSubmissionMetadata.model_validate(raw)
    except ValidationError as exc:
        raise AgentHandoffIngestionError(f"invalid submission.json: {exc}") from exc
    expected = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "submission_type": SUBMISSION_TYPE,
        "candidate_path": RELATIVE_CANDIDATE_PATH,
        "requested_action": REQUESTED_ACTION,
    }
    for field, value in expected.items():
        if getattr(metadata, field) != value:
            raise AgentHandoffIngestionError(
                f"invalid submission.json: {field} must be {value!r} (got {getattr(metadata, field)!r})"
            )
    if metadata.candidate_id != source_dir.name:
        raise AgentHandoffIngestionError(
            f"invalid submission.json: candidate_id must match submission directory '{source_dir.name}'"
        )
    return metadata


def _resolve_source_candidate(source_dir: Path, metadata: CandidateSubmissionMetadata) -> Path:
    candidate_path = Path(metadata.candidate_path)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise AgentHandoffIngestionError("invalid submission.json: candidate_path must be a relative child path")
    resolved = source_dir / candidate_path
    if not resolved.is_dir():
        raise AgentHandoffIngestionError(f"submission candidate_path is not a directory: {resolved}")
    return resolved


def _append_candidate_to_experiment_index(index_path: Path, candidate_id: str, description: str | None) -> None:
    if not index_path.is_file():
        raise AgentHandoffIngestionError(f"missing Experiment Index: {index_path}")
    text = index_path.read_text()
    candidate_ref = f"`candidates/{candidate_id}`"
    if candidate_ref in text:
        return
    row = (
        f"| {candidate_ref} | [`README.md`](candidates/{candidate_id}/README.md)"
        f"{_description_suffix(description)} | Pending full GVCCS Research Note. | Pending GVCCS Run. | "
        "Pending Harness Run; ingested from Agent Workspace. |\n"
    )
    marker = "## Chronological Research Notes"
    if marker in text:
        text = text.replace(marker, row + "\n" + marker, 1)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += row
    index_path.write_text(text)


def _append_research_note_to_experiment_index(index_path: Path, note_path: str, title: str) -> None:
    if not index_path.is_file():
        raise AgentHandoffIngestionError(f"missing Experiment Index: {index_path}")
    text = index_path.read_text()
    link = f"[`{title}`]({note_path})"
    if link in text or f"]({note_path})" in text:
        return
    entry = f"- {link}\n"
    marker = "## Chronological Research Notes"
    marker_index = text.find(marker)
    if marker_index == -1:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + marker + "\n" + entry
        index_path.write_text(text)
        return

    section_start = text.find("\n", marker_index)
    if section_start == -1:
        text += "\n" + entry
        index_path.write_text(text)
        return
    next_section = text.find("\n## ", section_start + 1)
    if next_section == -1:
        if not text.endswith("\n"):
            text += "\n"
        text += entry
    else:
        insertion = next_section + 1
        text = text[:insertion] + entry + text[insertion:]
    index_path.write_text(text)


def _research_note_title(note_text: str, note_path: Path) -> str:
    for line in note_text.splitlines():
        if line.startswith("# "):
            title = " ".join(line[2:].strip().split())
            if title:
                return title
    return note_path.stem.replace("-", " ").replace("_", " ").title()


def _description_suffix(description: str | None) -> str:
    if not description:
        return ""
    clean = " ".join(description.split())
    return f" — {clean}"


def _research_note_marker_path(note_path: Path) -> Path:
    return _file_marker_path(note_path)


def _file_marker_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{INGESTION_MARKER}")


def _has_valid_ingestion_marker(marker_path: Path, handoff_type: str) -> bool:
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("status") == "ingested"
        and marker.get("handoff_type") == handoff_type
        and isinstance(marker.get("canonical_path"), str)
        and bool(marker.get("canonical_path"))
    )


def _write_file_ingestion_marker_last(source: Path, handoff_type: str, artifact_id: str, canonical_path: str) -> None:
    marker = {
        "status": "ingested",
        "handoff_type": handoff_type,
        "artifact_id": artifact_id,
        "canonical_path": canonical_path,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    _file_marker_path(source).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")


def _write_research_note_ingestion_marker_last(source_note: Path, canonical_path: str) -> None:
    marker = {
        "status": "ingested",
        "handoff_type": "research_note",
        "note_path": canonical_path,
        "canonical_path": canonical_path,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    _research_note_marker_path(source_note).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")


def _write_ingestion_marker_last(source_dir: Path, candidate_id: str, canonical_path: str) -> None:
    marker = {
        "status": "ingested",
        "handoff_type": "candidate_submission",
        "candidate_id": candidate_id,
        "canonical_path": canonical_path,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (source_dir / INGESTION_MARKER).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
