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


def _description_suffix(description: str | None) -> str:
    if not description:
        return ""
    clean = " ".join(description.split())
    return f" — {clean}"


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
