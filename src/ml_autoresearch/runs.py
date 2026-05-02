"""Run lifecycle for local Candidate Experiment submissions."""

from __future__ import annotations

import json
import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.smoke import SmokeTestError, smoke_test_run


class RunStatus(StrEnum):
    """Reserved Run status vocabulary."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SMOKE_TESTING = "smoke_testing"
    SMOKE_FAILED = "smoke_failed"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunSubmission:
    run_id: str
    run_dir: Path
    status: RunStatus
    rejection_reason: str | None = None


def submit_candidate(candidate_dir: str | Path, runs_root: str | Path) -> RunSubmission:
    """Submit a local Candidate Experiment directory and create a Run record.

    Validation and smoke failures are represented as rejected/smoke_failed Runs
    so humans and agents can inspect logs and metadata for repair feedback.
    """

    source = Path(candidate_dir)
    root = Path(runs_root)
    run_id = _generate_run_id(root)
    run_dir = root / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    validation_log = logs_dir / "validation.log"

    created_at = _now_iso()

    try:
        manifest = validate_candidate_directory(source)
    except CandidateValidationError as exc:
        reason = str(exc)
        validation_log.write_text(f"Candidate validation failed: {reason}\n")
        _write_metadata(
            run_dir,
            run_id=run_id,
            status=RunStatus.REJECTED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=source,
            rejection_reason=reason,
            smoke_failure_reason=None,
        )
        return RunSubmission(run_id, run_dir, RunStatus.REJECTED, reason)

    validation_log.write_text("Candidate validation accepted.\n")
    shutil.copytree(source, run_dir / "candidate")
    _write_yaml(run_dir / "resolved_manifest.yaml", manifest.model_dump(mode="json"))
    _write_metadata(
        run_dir,
        run_id=run_id,
        status=RunStatus.SMOKE_TESTING,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=source,
        rejection_reason=None,
        smoke_failure_reason=None,
    )

    try:
        smoke_test_run(run_dir)
    except SmokeTestError as exc:
        reason = str(exc)
        _write_metadata(
            run_dir,
            run_id=run_id,
            status=RunStatus.SMOKE_FAILED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=source,
            rejection_reason=None,
            smoke_failure_reason=reason,
        )
        return RunSubmission(run_id, run_dir, RunStatus.SMOKE_FAILED, reason)

    _write_metadata(
        run_dir,
        run_id=run_id,
        status=RunStatus.ACCEPTED,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=source,
        rejection_reason=None,
        smoke_failure_reason=None,
    )
    return RunSubmission(run_id, run_dir, RunStatus.ACCEPTED)


def _generate_run_id(runs_root: Path) -> str:
    runs_root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)
        run_id = f"run_{timestamp}_{suffix}"
        if not (runs_root / run_id).exists():
            return run_id


def _write_metadata(
    run_dir: Path,
    *,
    run_id: str,
    status: RunStatus,
    created_at: str,
    updated_at: str,
    candidate_source: Path,
    rejection_reason: str | None,
    smoke_failure_reason: str | None,
) -> None:
    metadata = {
        "run_id": run_id,
        "status": status.value,
        "created_at": created_at,
        "updated_at": updated_at,
        "candidate_source": {"path": str(candidate_source.resolve())},
        "harness": {"package_version": _package_version()},
        "reserved_statuses": [member.value for member in RunStatus],
        "rejection_reason": rejection_reason,
        "smoke_failure_reason": smoke_failure_reason,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _package_version() -> str | None:
    try:
        return version("ml-autoresearch")
    except PackageNotFoundError:
        return None
