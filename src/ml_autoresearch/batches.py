"""Synchronous bounded Experiment Batch execution."""

from __future__ import annotations

import json
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.execution import ExecutionBackend
from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, record_research_event
from ml_autoresearch.runs import RunSubmission, RunStatus, get_run_summary, run_candidate_with_synthetic_fixture

MAX_BATCH_SIZE = 4
MAX_PARALLEL_RUNS = 4


class ExperimentBatchError(ValueError):
    """Raised when an Experiment Batch cannot be validated or executed."""


def run_experiment_batch_with_synthetic_fixture(
    batch_dir: str | Path,
    *,
    batches_root: str | Path,
    runs_root: str | Path,
    backend: ExecutionBackend | None = None,
    max_parallel_runs: int = MAX_PARALLEL_RUNS,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    ledger_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate and synchronously execute a bounded Experiment Batch.

    Static validation is all-or-nothing before any Run directory is created. Once
    execution starts, each Candidate Experiment gets its own Run and failures do
    not cancel sibling Runs.
    """

    batch_path = Path(batch_dir)
    candidates = _validate_batch_directory(batch_path)
    runs_root_path = Path(runs_root)
    batches_root_path = Path(batches_root)
    resolved_ledger_path = Path(ledger_path) if ledger_path is not None else batches_root_path.parent / CANONICAL_RESEARCH_LEDGER
    batch_id = _generate_batch_id(batches_root_path)
    batch_artifact_dir = batches_root_path / batch_id
    batch_artifact_dir.mkdir(parents=True)
    created_at = _now_iso()

    proposal_copy = batch_artifact_dir / "BATCH_PROPOSAL.md"
    proposal_copy.write_text((batch_path / "BATCH_PROPOSAL.md").read_text())
    _write_batch_metadata(
        batch_artifact_dir,
        batch_id=batch_id,
        status="running",
        created_at=created_at,
        updated_at=_now_iso(),
        source_path=batch_path,
        proposal_path=proposal_copy,
        max_parallel_runs=min(max_parallel_runs, MAX_PARALLEL_RUNS),
        candidates=[{"candidate_id": manifest.name, "source_path": str(path.resolve())} for path, manifest in candidates],
        runs=[],
    )
    record_research_event(
        "experiment_batch_created",
        {
            "batch_id": batch_id,
            "batch_path": str(batch_artifact_dir),
            "proposal_path": str(proposal_copy),
            "candidate_count": len(candidates),
        },
        ledger_path=resolved_ledger_path,
    )

    worker_count = max(1, min(max_parallel_runs, MAX_PARALLEL_RUNS, len(candidates)))
    run_records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_candidate = {
            executor.submit(
                run_candidate_with_synthetic_fixture,
                candidate_path,
                runs_root_path,
                max_prediction_samples=max_prediction_samples,
                prediction_sample_policy=prediction_sample_policy,
                backend=backend,
                ledger_path=resolved_ledger_path,
                require_proposal=False,
            ): (candidate_path, manifest.name)
            for candidate_path, manifest in candidates
        }
        for future in as_completed(future_to_candidate):
            candidate_path, candidate_id = future_to_candidate[future]
            try:
                run = future.result()
            except Exception as exc:  # noqa: BLE001 - a started sibling failure must not cancel the batch.
                run_records.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_path": str(candidate_path.resolve()),
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
                continue
            _tag_run_with_batch(run.run_dir, batch_id=batch_id, candidate_id=candidate_id)
            summary = get_run_summary(runs_root_path, run.run_id)
            run_records.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_path": str(candidate_path.resolve()),
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir),
                    "status": run.status.value,
                    "failure_classification": run.failure_classification.value if run.failure_classification is not None else summary.get("failure_classification"),
                    "reason": run.rejection_reason or summary.get("reason"),
                    "metrics": summary.get("metrics"),
                    "best_metrics": summary.get("best_metrics"),
                }
            )

    run_records.sort(key=lambda item: str(item.get("candidate_id") or ""))
    status = _batch_status(run_records)
    _write_batch_metadata(
        batch_artifact_dir,
        batch_id=batch_id,
        status=status,
        created_at=created_at,
        updated_at=_now_iso(),
        source_path=batch_path,
        proposal_path=proposal_copy,
        max_parallel_runs=worker_count,
        candidates=[{"candidate_id": manifest.name, "source_path": str(path.resolve())} for path, manifest in candidates],
        runs=run_records,
    )
    record_research_event(
        "experiment_batch_completed",
        {"batch_id": batch_id, "status": status, "run_count": sum(1 for item in run_records if item.get("run_id"))},
        ledger_path=resolved_ledger_path,
    )
    return get_batch_summary(batches_root_path, batch_id)


def list_batches(batches_root: str | Path) -> list[dict[str, object]]:
    """List local Experiment Batch artifact summaries."""

    root = Path(batches_root)
    if not root.exists():
        return []
    return sorted((_read_batch_summary_dir(path) for path in root.iterdir() if path.is_dir()), key=lambda item: str(item.get("batch_id") or ""))


def get_batch_summary(batches_root: str | Path, batch_id: str) -> dict[str, object]:
    """Read one local Experiment Batch summary."""

    batch_dir = Path(batches_root) / batch_id
    if not batch_dir.exists():
        return {"batch_id": batch_id, "batch_dir": str(batch_dir), "status": "missing", "error": "batch directory does not exist"}
    if not batch_dir.is_dir():
        return {"batch_id": batch_id, "batch_dir": str(batch_dir), "status": "corrupt", "error": "batch path is not a directory"}
    return _read_batch_summary_dir(batch_dir)


def _validate_batch_directory(batch_path: Path) -> list[tuple[Path, Any]]:
    if not batch_path.is_dir():
        raise ExperimentBatchError(f"Experiment Batch directory does not exist: {batch_path}")
    proposal = batch_path / "BATCH_PROPOSAL.md"
    if not proposal.is_file() or not proposal.read_text().strip():
        raise ExperimentBatchError("Experiment Batch requires a non-empty BATCH_PROPOSAL.md")
    candidates_root = batch_path / "candidates"
    if not candidates_root.is_dir():
        raise ExperimentBatchError("Experiment Batch requires a candidates/ directory")
    candidate_dirs = sorted(path for path in candidates_root.iterdir() if path.is_dir())
    if not candidate_dirs:
        raise ExperimentBatchError("Experiment Batch requires at least one Candidate Experiment")
    if len(candidate_dirs) > MAX_BATCH_SIZE:
        raise ExperimentBatchError(f"Experiment Batch may contain at most {MAX_BATCH_SIZE} Candidate Experiments")

    validated: list[tuple[Path, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for candidate_dir in candidate_dirs:
        try:
            manifest = validate_candidate_directory(candidate_dir, require_proposal=False, require_readme=True)
            if manifest.name != candidate_dir.name:
                errors.append(f"{candidate_dir.name}: manifest name must match candidate directory (got {manifest.name!r})")
            elif manifest.name in seen:
                errors.append(f"{candidate_dir.name}: duplicate candidate id {manifest.name!r}")
            else:
                seen.add(manifest.name)
                validated.append((candidate_dir, manifest))
        except CandidateValidationError as exc:
            errors.append(f"{candidate_dir.name}: {exc}")
    if errors:
        raise ExperimentBatchError("Experiment Batch static validation failed: " + "; ".join(errors))
    return validated


def _batch_status(run_records: list[dict[str, object]]) -> str:
    statuses = {str(item.get("status")) for item in run_records}
    if statuses == {RunStatus.COMPLETED.value}:
        return "completed"
    if RunStatus.COMPLETED.value in statuses:
        return "partially_failed"
    return "failed"


def _tag_run_with_batch(run_dir: Path, *, batch_id: str, candidate_id: str) -> None:
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["batch_id"] = batch_id
    metadata["batch_candidate_id"] = candidate_id
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _write_batch_metadata(
    batch_dir: Path,
    *,
    batch_id: str,
    status: str,
    created_at: str,
    updated_at: str,
    source_path: Path,
    proposal_path: Path,
    max_parallel_runs: int,
    candidates: list[dict[str, object]],
    runs: list[dict[str, object]],
) -> None:
    payload = {
        "schema_version": "experiment_batch.v1",
        "batch_id": batch_id,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "source_path": str(source_path.resolve()),
        "proposal_path": str(proposal_path),
        "max_batch_size": MAX_BATCH_SIZE,
        "max_parallel_runs": max_parallel_runs,
        "candidates": candidates,
        "runs": runs,
    }
    (batch_dir / "batch_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_batch_summary_dir(batch_dir: Path) -> dict[str, object]:
    metadata_path = batch_dir / "batch_metadata.json"
    if not metadata_path.exists():
        return {"batch_id": batch_dir.name, "batch_dir": str(batch_dir), "status": "missing_metadata", "error": "batch_metadata.json is missing"}
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"batch_id": batch_dir.name, "batch_dir": str(batch_dir), "status": "corrupt", "error": f"cannot read batch_metadata.json: {exc}"}
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata["batch_dir"] = str(batch_dir)
        return metadata
    return {"batch_id": batch_dir.name, "batch_dir": str(batch_dir), "status": "corrupt", "error": "batch_metadata.json is not an object"}


def _generate_batch_id(batches_root: Path) -> str:
    batches_root.mkdir(parents=True, exist_ok=True)
    while True:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)
        batch_id = f"batch_{timestamp}_{suffix}"
        if not (batches_root / batch_id).exists():
            return batch_id


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
