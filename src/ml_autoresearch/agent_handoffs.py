"""Harness-owned Agent Handoff Ingestion."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, record_research_event
from ml_autoresearch.research_notes import ResearchNoteFigureError, validate_research_note_figure_provenance
from ml_autoresearch.submissions import (
    RELATIVE_CANDIDATE_PATH,
    REQUESTED_ACTION,
    SUBMISSION_SCHEMA_VERSION,
    SUBMISSION_TYPE,
)

INGESTION_MARKER = "INGESTED.json"


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


def _discover_one_uningested_submission(submissions_root: Path) -> Path:
    if not submissions_root.is_dir():
        raise AgentHandoffIngestionError(f"Candidate Submission Queue does not exist: {submissions_root}")
    submissions = sorted(
        path for path in submissions_root.iterdir() if path.is_dir() and not (path / INGESTION_MARKER).exists()
    )
    if len(submissions) != 1:
        raise AgentHandoffIngestionError(
            f"expected exactly one un-ingested Candidate Submission in {submissions_root}, found {len(submissions)}"
        )
    return submissions[0]


def _discover_one_uningested_research_note(notes_root: Path) -> Path:
    if not notes_root.is_dir():
        raise AgentHandoffIngestionError(f"Agent Workspace Research Notes directory does not exist: {notes_root}")
    notes = sorted(
        path
        for path in notes_root.glob("*.md")
        if path.is_file() and not _research_note_marker_path(path).exists()
    )
    if len(notes) != 1:
        raise AgentHandoffIngestionError(
            f"expected exactly one un-ingested Research Note in {notes_root}, found {len(notes)}"
        )
    return notes[0]


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
    return note_path.with_name(f"{note_path.name}.{INGESTION_MARKER}")


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
