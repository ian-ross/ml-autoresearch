"""Run lifecycle for local Candidate Experiment submissions."""

from __future__ import annotations

import json
import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.execution import DockerOperationTimeoutError, ExecutionBackend, NativeBackend, backend_metadata
from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, record_research_event
from ml_autoresearch.smoke import SmokeTestError
from ml_autoresearch.gvccs import GVCCSDataError
from ml_autoresearch.training import TrainingError


class RunStatus(StrEnum):
    """Reserved Run status vocabulary."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SMOKE_TESTING = "smoke_testing"
    SMOKE_FAILED = "smoke_failed"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class RunFailureClassification(StrEnum):
    """Approved Run Failure Classification vocabulary."""

    CANDIDATE_BUG = "candidate_bug"
    CONTRACT_VIOLATION = "contract_violation"
    RESOURCE_FAILURE = "resource_failure"
    HARNESS_FAILURE = "harness_failure"
    BAD_RESEARCH_RESULT = "bad_research_result"
    UNKNOWN = "unknown"


MAX_RESOURCE_RETRY_ATTEMPTS = 3
_RESOURCE_FAILURE_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "hip out of memory",
    "mps backend out of memory",
    "defaultcpuallocator: can't allocate memory",
    "cannot allocate memory",
    "memoryerror",
    "std::bad_alloc",
)


def validate_run_failure_classification(value: str | RunFailureClassification | None) -> RunFailureClassification | None:
    """Validate a Run Failure Classification value from metadata or callers."""

    if value is None:
        return None
    try:
        return RunFailureClassification(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in RunFailureClassification)
        raise ValueError(f"invalid run failure classification '{value}'; expected one of: {allowed}") from exc


def is_resource_failure(exc: BaseException | str) -> bool:
    """Classify GPU/compute exhaustion messages as Resource Failures."""

    text = str(exc).lower()
    return any(marker in text for marker in _RESOURCE_FAILURE_MARKERS)


@dataclass(frozen=True)
class RunSubmission:
    run_id: str
    run_dir: Path
    status: RunStatus
    rejection_reason: str | None = None
    failure_classification: RunFailureClassification | None = None


def run_candidate_with_synthetic_fixture(
    candidate_dir: str | Path,
    runs_root: str | Path,
    *,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
) -> RunSubmission:
    """Validate, smoke-test, and synchronously train a Candidate Experiment Run."""

    return _run_candidate_synthetic_training(
        candidate_dir,
        runs_root,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
        backend=backend,
        ledger_path=ledger_path,
        require_proposal=require_proposal,
    )


def run_candidate_with_gvccs_data(
    candidate_dir: str | Path,
    runs_root: str | Path,
    data_root: str | Path,
    *,
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
) -> RunSubmission:
    """Validate, smoke-test, and synchronously train a Candidate Experiment Run on local GVCCS data."""

    data_path = _validate_host_data_root(data_root)
    selected_backend = backend or NativeBackend()
    return _run_candidate_training(
        candidate_dir,
        runs_root,
        lambda run_dir: selected_backend.train_gvccs(
            run_dir,
            data_path,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        ),
        backend=selected_backend,
        dataset=_gvccs_dataset_metadata(data_path),
        ledger_path=ledger_path,
        require_proposal=require_proposal,
    )


def _run_candidate_synthetic_training(
    candidate_dir: str | Path,
    runs_root: str | Path,
    *,
    max_prediction_samples: int,
    prediction_sample_policy: str,
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
) -> RunSubmission:
    resolved_ledger_path = _resolve_ledger_path(runs_root, ledger_path)
    run = submit_candidate(
        candidate_dir,
        runs_root,
        backend=backend,
        ledger_path=resolved_ledger_path,
        require_proposal=require_proposal,
    )
    if run.status != RunStatus.ACCEPTED:
        return run

    metadata = _read_metadata(run.run_dir)
    created_at = str(metadata["created_at"])
    candidate_source = Path(str(metadata["candidate_source"]["path"]))
    candidate_id = _candidate_id_from_run_dir(run.run_dir)
    record_research_event(
        "run_started",
        {"run_id": run.run_id, "candidate_id": candidate_id},
        ledger_path=resolved_ledger_path,
    )
    execution_backend_metadata = metadata.get("execution_backend")
    repair_lineage = metadata.get("repair_lineage") if isinstance(metadata.get("repair_lineage"), dict) else None
    _write_metadata(
        run.run_dir,
        run_id=run.run_id,
        status=RunStatus.TRAINING,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=candidate_source,
        rejection_reason=None,
        smoke_failure_reason=None,
        training_failure_reason=None,
        execution_backend=execution_backend_metadata,
        repair_lineage=repair_lineage,
    )
    selected_backend = backend or NativeBackend()
    training_result = None
    resource_lifecycle = None
    try:
        training_result, resource_lifecycle = _run_with_resource_retries(
            run.run_dir,
            lambda: selected_backend.train_synthetic(
                run.run_dir,
                max_prediction_samples=max_prediction_samples,
                prediction_sample_policy=prediction_sample_policy,
            ),
        )
        artifacts = _validate_synthetic_training_outputs(run.run_dir)
    except DockerOperationTimeoutError as exc:
        reason = str(exc)
        _write_metadata(
            run.run_dir,
            run_id=run.run_id,
            status=RunStatus.FAILED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=candidate_source,
            rejection_reason=None,
            smoke_failure_reason=None,
            training_failure_reason=reason,
            failure_classification=RunFailureClassification.RESOURCE_FAILURE,
            execution_backend=execution_backend_metadata,
            training_lifecycle={"status": "timeout_forced", "timeout": exc.timeout_metadata},
            repair_lineage=repair_lineage,
        )
        _record_run_failed(resolved_ledger_path, run.run_id, reason, RunFailureClassification.RESOURCE_FAILURE)
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason, RunFailureClassification.RESOURCE_FAILURE)
    except (TrainingError, RuntimeError) as exc:
        reason = str(exc)
        classification = RunFailureClassification.RESOURCE_FAILURE if is_resource_failure(exc) else RunFailureClassification.CANDIDATE_BUG
        failure_lifecycle = exc.lifecycle if isinstance(exc, ResourceRetryExhaustedError) else None
        _write_metadata(
            run.run_dir,
            run_id=run.run_id,
            status=RunStatus.FAILED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=candidate_source,
            rejection_reason=None,
            smoke_failure_reason=None,
            training_failure_reason=reason,
            failure_classification=classification,
            execution_backend=execution_backend_metadata,
            training_lifecycle=failure_lifecycle,
            repair_lineage=repair_lineage,
        )
        _record_run_failed(resolved_ledger_path, run.run_id, reason, classification)
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason, classification)

    _write_metadata(
        run.run_dir,
        run_id=run.run_id,
        status=RunStatus.COMPLETED,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=candidate_source,
        rejection_reason=None,
        smoke_failure_reason=None,
        training_failure_reason=None,
        artifacts=artifacts,
        execution_backend=execution_backend_metadata,
        training_lifecycle=_merge_training_lifecycle(training_result, resource_lifecycle or {}),
        repair_lineage=repair_lineage,
    )
    _record_run_completed(resolved_ledger_path, run.run_id, run.run_dir)
    return RunSubmission(run.run_id, run.run_dir, RunStatus.COMPLETED)


def _run_candidate_training(
    candidate_dir: str | Path,
    runs_root: str | Path,
    trainer,
    *,
    backend: ExecutionBackend | None = None,
    dataset: dict[str, object] | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
) -> RunSubmission:
    resolved_ledger_path = _resolve_ledger_path(runs_root, ledger_path)
    run = submit_candidate(
        candidate_dir,
        runs_root,
        backend=backend,
        ledger_path=resolved_ledger_path,
        require_proposal=require_proposal,
    )
    if run.status != RunStatus.ACCEPTED:
        return run

    metadata = _read_metadata(run.run_dir)
    created_at = str(metadata["created_at"])
    candidate_source = Path(str(metadata["candidate_source"]["path"]))
    candidate_id = _candidate_id_from_run_dir(run.run_dir)
    record_research_event(
        "run_started",
        {"run_id": run.run_id, "candidate_id": candidate_id},
        ledger_path=resolved_ledger_path,
    )
    repair_lineage = metadata.get("repair_lineage") if isinstance(metadata.get("repair_lineage"), dict) else None
    _write_metadata(
        run.run_dir,
        run_id=run.run_id,
        status=RunStatus.TRAINING,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=candidate_source,
        rejection_reason=None,
        smoke_failure_reason=None,
        training_failure_reason=None,
        execution_backend=metadata.get("execution_backend"),
        dataset=dataset,
        repair_lineage=repair_lineage,
    )
    resource_lifecycle = None
    try:
        training_result, resource_lifecycle = _run_with_resource_retries(run.run_dir, lambda: trainer(run.run_dir))
    except DockerOperationTimeoutError as exc:
        reason = str(exc)
        _write_training_failure_log(run.run_dir, reason)
        _write_metadata(
            run.run_dir,
            run_id=run.run_id,
            status=RunStatus.FAILED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=candidate_source,
            rejection_reason=None,
            smoke_failure_reason=None,
            training_failure_reason=reason,
            failure_classification=RunFailureClassification.RESOURCE_FAILURE,
            execution_backend=metadata.get("execution_backend"),
            dataset=dataset,
            training_lifecycle={"status": "timeout_forced", "timeout": exc.timeout_metadata},
            repair_lineage=repair_lineage,
        )
        _record_run_failed(resolved_ledger_path, run.run_id, reason, RunFailureClassification.RESOURCE_FAILURE)
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason, RunFailureClassification.RESOURCE_FAILURE)
    except (TrainingError, RuntimeError, GVCCSDataError) as exc:
        reason = str(exc)
        classification = (
            RunFailureClassification.HARNESS_FAILURE
            if isinstance(exc, GVCCSDataError)
            else RunFailureClassification.RESOURCE_FAILURE if is_resource_failure(exc) else RunFailureClassification.CANDIDATE_BUG
        )
        failure_lifecycle = exc.lifecycle if isinstance(exc, ResourceRetryExhaustedError) else None
        _write_training_failure_log(run.run_dir, reason)
        _write_metadata(
            run.run_dir,
            run_id=run.run_id,
            status=RunStatus.FAILED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=candidate_source,
            rejection_reason=None,
            smoke_failure_reason=None,
            training_failure_reason=reason,
            failure_classification=classification,
            execution_backend=metadata.get("execution_backend"),
            dataset=dataset,
            training_lifecycle=failure_lifecycle,
            repair_lineage=repair_lineage,
        )
        _record_run_failed(resolved_ledger_path, run.run_id, reason, classification)
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason, classification)

    _write_metadata(
        run.run_dir,
        run_id=run.run_id,
        status=RunStatus.COMPLETED,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=candidate_source,
        rejection_reason=None,
        smoke_failure_reason=None,
        training_failure_reason=None,
        artifacts=_artifacts_from_training_result(training_result),
        execution_backend=metadata.get("execution_backend"),
        dataset=dataset,
        training_lifecycle=_merge_training_lifecycle(training_result, resource_lifecycle or {}),
        repair_lineage=repair_lineage,
    )
    _record_run_completed(resolved_ledger_path, run.run_id, run.run_dir)
    return RunSubmission(run.run_id, run.run_dir, RunStatus.COMPLETED)


def list_runs(runs_root: str | Path) -> list[dict[str, object]]:
    """Read summaries for all local Run artifact directories under ``runs_root``.

    This observes only files already present under the local ``runs/`` tree and
    never requires MLflow. Corrupt or incomplete Run directories are returned as
    explicit summary records so humans and agents can see what was skipped.
    """

    root = Path(runs_root)
    if not root.exists():
        return []
    summaries = [_read_run_summary_dir(run_dir) for run_dir in root.iterdir() if run_dir.is_dir()]
    return sorted(summaries, key=lambda item: str(item.get("run_id") or ""))


def get_run_summary(runs_root: str | Path, run_id: str) -> dict[str, object]:
    """Read one local Run summary from ``runs_root/run_id``."""

    run_dir = Path(runs_root) / run_id
    if not run_dir.exists():
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "missing", "error": "run directory does not exist"}
    if not run_dir.is_dir():
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "corrupt", "error": "run path is not a directory"}
    return _read_run_summary_dir(run_dir)


def get_best_runs(runs_root: str | Path, *, metric: str = "val/dice", limit: int | None = None) -> list[dict[str, object]]:
    """Return completed local Runs ranked descending by a metric, ``val/dice`` by default."""

    ranked: list[dict[str, object]] = []
    for summary in list_runs(runs_root):
        best_metrics = summary.get("best_metrics")
        best_metric_name = best_metrics.get("selection_metric") if isinstance(best_metrics, dict) else None
        if isinstance(best_metrics, dict) and best_metric_name == metric:
            value = best_metrics.get("selection_value")
        else:
            metrics = summary.get("metrics")
            value = metrics.get(metric) if isinstance(metrics, dict) else None
        if summary.get("status") == RunStatus.COMPLETED.value and isinstance(value, int | float):
            ranked_summary = dict(summary)
            ranked_summary["rank_metric_name"] = metric
            ranked_summary["rank_metric"] = float(value)
            ranked.append(ranked_summary)
    ranked.sort(key=lambda item: float(item["rank_metric"]), reverse=True)
    if limit is not None:
        return ranked[:limit]
    return ranked


def _read_run_summary_dir(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {"run_id": run_dir.name, "run_dir": str(run_dir), "status": "missing_metadata", "error": "run_metadata.json is missing"}
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:  # noqa: BLE001 - observation must report corrupt artifacts clearly.
        return {"run_id": run_dir.name, "run_dir": str(run_dir), "status": "corrupt", "error": f"cannot read run_metadata.json: {exc}"}

    run_id = str(metadata.get("run_id") or run_dir.name)
    status = str(metadata.get("status") or "unknown")
    summary: dict[str, object] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "candidate_source": metadata.get("candidate_source"),
        "evaluations": _read_evaluation_summaries(run_dir),
    }
    reason = metadata.get("rejection_reason") or metadata.get("smoke_failure_reason") or metadata.get("training_failure_reason")
    if reason is not None:
        summary["reason"] = reason
    if metadata.get("failure_classification") is not None:
        summary["failure_classification"] = metadata["failure_classification"]
    if "artifacts" in metadata:
        summary["artifacts"] = metadata["artifacts"]

    outputs_dir = _outputs_dir(run_dir)
    best_metrics_path = outputs_dir / "best_metrics.json"
    if best_metrics_path.exists():
        try:
            summary["best_metrics"] = json.loads(best_metrics_path.read_text())
        except Exception as exc:  # noqa: BLE001
            summary["best_metrics_error"] = f"cannot read best_metrics.json: {exc}"

    final_metrics_path = outputs_dir / "final_metrics.json"
    if final_metrics_path.exists():
        try:
            metrics = json.loads(final_metrics_path.read_text())
            summary["metrics"] = metrics
            if isinstance(metrics, dict) and "artifacts" in metrics and "artifacts" not in summary:
                summary["artifacts"] = metrics["artifacts"]
        except Exception as exc:  # noqa: BLE001
            summary["metrics_error"] = f"cannot read final_metrics.json: {exc}"
    elif status == RunStatus.COMPLETED.value:
        summary["metrics_error"] = "final_metrics.json is missing"
    return summary


def _read_evaluation_summaries(run_dir: Path) -> list[dict[str, object]]:
    evaluations_dir = _outputs_dir(run_dir) / "evaluations"
    if not evaluations_dir.is_dir():
        return []
    return [_read_evaluation_summary_dir(evaluation_dir) for evaluation_dir in sorted(evaluations_dir.iterdir()) if evaluation_dir.is_dir()]


def _read_evaluation_summary_dir(evaluation_dir: Path) -> dict[str, object]:
    metadata_path = evaluation_dir / "evaluation_metadata.json"
    if not metadata_path.exists():
        return {
            "evaluation_id": evaluation_dir.name,
            "path": str(evaluation_dir),
            "status": "missing_metadata",
            "error": "evaluation_metadata.json is missing",
        }
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:  # noqa: BLE001 - observation should not corrupt the whole Run summary.
        return {
            "evaluation_id": evaluation_dir.name,
            "path": str(evaluation_dir),
            "status": "corrupt",
            "error": f"cannot read evaluation_metadata.json: {exc}",
        }
    if not isinstance(metadata, dict):
        return {
            "evaluation_id": evaluation_dir.name,
            "path": str(evaluation_dir),
            "status": "corrupt",
            "error": "evaluation_metadata.json is not a JSON object",
        }

    evaluation_id = metadata.get("evaluation_id") if isinstance(metadata.get("evaluation_id"), str) else evaluation_dir.name
    summary: dict[str, object] = {
        "evaluation_id": evaluation_id,
        "status": metadata.get("status", "unknown"),
        "mode": metadata.get("mode"),
        "path": str(evaluation_dir),
    }
    created_at = metadata.get("created_at") or metadata.get("started_at")
    if created_at is not None:
        summary["created_at"] = created_at
    completed_at = metadata.get("completed_at") or metadata.get("failed_at")
    if completed_at is not None:
        summary["completed_at"] = completed_at
    failure_reason = metadata.get("failure_reason")
    if failure_reason is not None:
        summary["failure_reason"] = failure_reason
    return summary


def _resolved_manifest_payload(manifest: object) -> dict[str, object]:
    payload = manifest.model_dump(mode="json")
    data_policy = payload.setdefault("data", {})
    selected = data_policy.get("augmentation_policy", "none")
    data_policy["augmentation_policy"] = selected
    data_policy["augmentation_policy_effective"] = selected
    return payload


def _repair_lineage_from_manifest(manifest: object) -> dict[str, object] | None:
    repair = getattr(manifest, "repair", None)
    if repair is None:
        return None
    return repair.model_dump(mode="json")


def _repair_count_for_original_proposal(ledger_path: Path, original_proposal_id: str) -> int:
    if not ledger_path.exists():
        return 0
    count = 0
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        repair_lineage = event.get("repair_lineage") if event.get("event_type") == "candidate_created" else None
        if isinstance(repair_lineage, dict) and repair_lineage.get("original_proposal_id") == original_proposal_id:
            count += 1
    return count


def _enforce_autonomous_repair_limit(repair_lineage: dict[str, object] | None, ledger_path: Path, *, require_proposal: bool) -> str | None:
    if repair_lineage is None or not require_proposal:
        return None
    original_proposal_id = str(repair_lineage["original_proposal_id"])
    if _repair_count_for_original_proposal(ledger_path, original_proposal_id) >= 2:
        return f"autonomous-mode permits at most two Repair Candidates per original proposal: {original_proposal_id}"
    return None



def submit_candidate(
    candidate_dir: str | Path,
    runs_root: str | Path,
    *,
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
) -> RunSubmission:
    """Submit a local Candidate Experiment directory and create a Run record.

    Validation and smoke failures are represented as rejected/smoke_failed Runs
    so humans and agents can inspect logs and metadata for repair feedback.
    """

    source = Path(candidate_dir)
    execution_backend = backend or NativeBackend()
    execution_backend_metadata = backend_metadata(execution_backend)
    root = Path(runs_root)
    resolved_ledger_path = _resolve_ledger_path(runs_root, ledger_path)
    run_id = _generate_run_id(root)
    run_dir = root / run_id
    logs_dir = _outputs_dir(run_dir) / "logs"
    logs_dir.mkdir(parents=True)
    validation_log = logs_dir / "validation.log"

    created_at = _now_iso()

    try:
        manifest = validate_candidate_directory(source, require_proposal=require_proposal)
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
            training_failure_reason=None,
            failure_classification=RunFailureClassification.CONTRACT_VIOLATION,
            execution_backend=execution_backend_metadata,
        )
        return RunSubmission(run_id, run_dir, RunStatus.REJECTED, reason, RunFailureClassification.CONTRACT_VIOLATION)

    repair_lineage = _repair_lineage_from_manifest(manifest)
    repair_policy_reason = _enforce_autonomous_repair_limit(
        repair_lineage,
        resolved_ledger_path,
        require_proposal=require_proposal,
    )
    if repair_policy_reason is not None:
        validation_log.write_text(f"Candidate validation failed: {repair_policy_reason}\n")
        _write_metadata(
            run_dir,
            run_id=run_id,
            status=RunStatus.REJECTED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=source,
            rejection_reason=repair_policy_reason,
            smoke_failure_reason=None,
            training_failure_reason=None,
            failure_classification=RunFailureClassification.CONTRACT_VIOLATION,
            execution_backend=execution_backend_metadata,
            repair_lineage=repair_lineage,
        )
        return RunSubmission(run_id, run_dir, RunStatus.REJECTED, repair_policy_reason, RunFailureClassification.CONTRACT_VIOLATION)

    validation_log.write_text("Candidate validation accepted.\n")
    proposal_path = source / "PROPOSAL.md"
    if proposal_path.is_file():
        _record_proposal_created_event(proposal_path, manifest.name, resolved_ledger_path=resolved_ledger_path)
    _record_candidate_created_event(source, manifest.name, proposal_id=manifest.name if proposal_path.is_file() else None, repair_lineage=repair_lineage, resolved_ledger_path=resolved_ledger_path)
    shutil.copytree(source, run_dir / "candidate")
    _write_yaml(run_dir / "resolved_manifest.yaml", _resolved_manifest_payload(manifest))
    _write_metadata(
        run_dir,
        run_id=run_id,
        status=RunStatus.SMOKE_TESTING,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=source,
        rejection_reason=None,
        smoke_failure_reason=None,
        training_failure_reason=None,
        execution_backend=execution_backend_metadata,
        repair_lineage=repair_lineage,
    )

    try:
        execution_backend.smoke_test(run_dir)
    except (SmokeTestError, RuntimeError) as exc:
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
            training_failure_reason=None,
            failure_classification=RunFailureClassification.CANDIDATE_BUG,
            execution_backend=execution_backend_metadata,
            repair_lineage=repair_lineage,
        )
        return RunSubmission(run_id, run_dir, RunStatus.SMOKE_FAILED, reason, RunFailureClassification.CANDIDATE_BUG)

    _write_metadata(
        run_dir,
        run_id=run_id,
        status=RunStatus.ACCEPTED,
        created_at=created_at,
        updated_at=_now_iso(),
        candidate_source=source,
        rejection_reason=None,
        smoke_failure_reason=None,
        training_failure_reason=None,
        execution_backend=execution_backend_metadata,
        repair_lineage=repair_lineage,
    )
    record_research_event(
        "candidate_submitted",
        {"candidate_id": manifest.name, "run_id": run_id},
        ledger_path=resolved_ledger_path,
    )
    return RunSubmission(run_id, run_dir, RunStatus.ACCEPTED)


def _outputs_dir(run_dir: Path) -> Path:
    return run_dir / "outputs"


def _run_with_resource_retries(run_dir: Path, operation: Callable[[], object]) -> tuple[object, dict[str, object]]:
    """Run Harness-owned training with bounded Resource Failure batch-size retries."""

    requested_batch_size = _resolved_manifest_batch_size(run_dir)
    effective_batch_size = requested_batch_size
    attempts: list[dict[str, object]] = []
    retry_count = 0
    while True:
        attempt_number = len(attempts) + 1
        _set_resolved_manifest_effective_batch_size(
            run_dir,
            requested_batch_size=requested_batch_size,
            effective_batch_size=effective_batch_size,
        )
        try:
            result = operation()
        except (TrainingError, RuntimeError) as exc:
            reason = str(exc)
            if not is_resource_failure(exc):
                raise
            attempt: dict[str, object] = {
                "attempt": attempt_number,
                "batch_size": effective_batch_size,
                "outcome": "resource_failure",
                "failure_classification": RunFailureClassification.RESOURCE_FAILURE.value,
                "reason": reason,
            }
            can_retry = retry_count < MAX_RESOURCE_RETRY_ATTEMPTS and effective_batch_size > 1
            if can_retry:
                next_batch_size = max(1, effective_batch_size // 2)
                attempt["next_batch_size"] = next_batch_size
                attempts.append(attempt)
                _append_resource_retry_log(
                    run_dir,
                    f"Resource Failure on attempt {attempt_number} with batch_size={effective_batch_size}: {reason}; retrying with batch_size={next_batch_size}.",
                )
                effective_batch_size = next_batch_size
                retry_count += 1
                continue
            attempts.append(attempt)
            _append_resource_retry_log(
                run_dir,
                f"Resource Failure retry exhausted after attempt {attempt_number} with batch_size={effective_batch_size}: {reason}.",
            )
            lifecycle = _resource_retry_lifecycle(
                requested_batch_size=requested_batch_size,
                effective_batch_size=effective_batch_size,
                attempts=attempts,
                exhausted=True,
            )
            raise ResourceRetryExhaustedError(reason, lifecycle=lifecycle) from exc
        attempts.append({"attempt": attempt_number, "batch_size": effective_batch_size, "outcome": "completed"})
        if retry_count:
            _append_resource_retry_log(
                run_dir,
                f"Resource Failure retry succeeded on attempt {attempt_number} with batch_size={effective_batch_size}.",
            )
        lifecycle = _resource_retry_lifecycle(
            requested_batch_size=requested_batch_size,
            effective_batch_size=effective_batch_size,
            attempts=attempts,
            exhausted=False,
        )
        return result, lifecycle


class ResourceRetryExhaustedError(TrainingError):
    """Raised when bounded Resource Failure retry attempts are exhausted."""

    def __init__(self, reason: str, *, lifecycle: dict[str, object]):
        super().__init__(f"Resource Failure retry exhausted: {reason}")
        self.lifecycle = lifecycle


def _resource_retry_lifecycle(
    *, requested_batch_size: int, effective_batch_size: int, attempts: list[dict[str, object]], exhausted: bool
) -> dict[str, object]:
    retry_count = sum(1 for attempt in attempts if attempt.get("outcome") == "resource_failure" and "next_batch_size" in attempt)
    status = "resource_retry_exhausted" if exhausted else ("completed_after_resource_retry" if retry_count else "completed")
    return {
        "status": status,
        "resource_retry": {
            "enabled": True,
            "max_retries": MAX_RESOURCE_RETRY_ATTEMPTS,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "retry_count": retry_count,
            "exhausted": exhausted,
            "attempts": attempts,
        },
    }


def _resolved_manifest_batch_size(run_dir: Path) -> int:
    manifest = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    return int(manifest["training"].get("batch_size_requested", manifest["training"]["batch_size"]))


def _set_resolved_manifest_effective_batch_size(
    run_dir: Path, *, requested_batch_size: int, effective_batch_size: int
) -> None:
    manifest_path = run_dir / "resolved_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["training"]["batch_size_requested"] = requested_batch_size
    manifest["training"]["batch_size_effective"] = effective_batch_size
    manifest["training"]["batch_size"] = effective_batch_size
    _write_yaml(manifest_path, manifest)


def _append_resource_retry_log(run_dir: Path, line: str) -> None:
    log_path = _outputs_dir(run_dir) / "logs" / "resource_retry.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(line + "\n")


def _resolve_ledger_path(runs_root: str | Path, ledger_path: str | Path | None) -> Path:
    if ledger_path is not None:
        return Path(ledger_path)
    return Path(runs_root).parent / CANONICAL_RESEARCH_LEDGER


def _candidate_id_from_run_dir(run_dir: Path) -> str:
    manifest_path = run_dir / "resolved_manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text())
    return str(data["name"])


def _record_proposal_created_event(proposal_path: Path, candidate_id: str, *, resolved_ledger_path: Path) -> None:
    record_research_event(
        "proposal_created",
        {
            "proposal_id": candidate_id,
            "proposal_path": str(proposal_path),
            "candidate_id": candidate_id,
        },
        ledger_path=resolved_ledger_path,
    )


def _record_candidate_created_event(
    candidate_path: Path,
    candidate_id: str,
    *,
    resolved_ledger_path: Path,
    proposal_id: str | None = None,
    repair_lineage: dict[str, object] | None = None,
) -> None:
    fields: dict[str, object] = {
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path),
    }
    if proposal_id is not None:
        fields["proposal_id"] = proposal_id
    if repair_lineage is not None:
        fields["repair_lineage"] = repair_lineage
    record_research_event("candidate_created", fields, ledger_path=resolved_ledger_path)


def _record_run_completed(ledger_path: Path, run_id: str, run_dir: Path) -> None:
    metrics_path = run_dir / "outputs" / "final_metrics.json"
    record_research_event(
        "run_completed",
        {"run_id": run_id, "metrics_path": str(metrics_path)},
        ledger_path=ledger_path,
    )


def _record_run_failed(
    ledger_path: Path,
    run_id: str,
    reason: str,
    failure_classification: RunFailureClassification = RunFailureClassification.UNKNOWN,
) -> None:
    record_research_event(
        "run_failed",
        {
            "run_id": run_id,
            "error": reason or "unknown failure",
            "failure_classification": failure_classification.value,
        },
        ledger_path=ledger_path,
    )


def _validate_host_data_root(data_root: str | Path) -> Path:
    path = Path(data_root)
    if not path.exists():
        raise GVCCSDataError(f"GVCCS data root does not exist: {path}")
    if not path.is_dir():
        raise GVCCSDataError(f"GVCCS data root is not a directory: {path}")
    return path.resolve()


def _gvccs_dataset_metadata(data_root: Path) -> dict[str, object]:
    return {"id": "gvccs", "host_data_path": str(data_root), "container_data_path": "/data"}


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
    training_failure_reason: str | None,
    failure_classification: str | RunFailureClassification | None = None,
    artifacts: dict[str, object] | None = None,
    execution_backend: object | None = None,
    dataset: dict[str, object] | None = None,
    training_lifecycle: dict[str, object] | None = None,
    repair_lineage: dict[str, object] | None = None,
) -> None:
    metadata = {
        "run_id": run_id,
        "status": status.value,
        "created_at": created_at,
        "updated_at": updated_at,
        "candidate_source": {"path": str(candidate_source.resolve())},
        "harness": {"package_version": _package_version()},
        "reserved_statuses": [member.value for member in RunStatus],
        "reserved_failure_classifications": [member.value for member in RunFailureClassification],
        "failure_classification": (
            validate_run_failure_classification(failure_classification).value
            if failure_classification is not None
            else None
        ),
        "rejection_reason": rejection_reason,
        "smoke_failure_reason": smoke_failure_reason,
        "training_failure_reason": training_failure_reason,
    }
    if execution_backend is not None:
        metadata["execution_backend"] = execution_backend
    if dataset is not None:
        metadata["dataset"] = dataset
    if artifacts is not None:
        metadata["artifacts"] = artifacts
    if training_lifecycle is not None:
        metadata["training_lifecycle"] = training_lifecycle
    if repair_lineage is not None:
        metadata["repair_lineage"] = repair_lineage
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _read_metadata(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run_metadata.json").read_text())


def _write_training_failure_log(run_dir: Path, reason: str) -> None:
    log_path = _outputs_dir(run_dir) / "logs" / "training.log"
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"Training failed: {reason}\n")


def _validate_synthetic_training_outputs(run_dir: Path) -> dict[str, object] | None:
    outputs_dir = _outputs_dir(run_dir)
    required = [
        outputs_dir / "metrics.jsonl",
        outputs_dir / "final_metrics.json",
        outputs_dir / "best_metrics.json",
        outputs_dir / "logs" / "training.log",
    ]
    for path in required:
        if not path.exists():
            raise TrainingError(f"required synthetic training artifact is missing: {path.relative_to(run_dir)}")
    try:
        final_metrics = json.loads((outputs_dir / "final_metrics.json").read_text())
    except Exception as exc:  # noqa: BLE001 - backend output validation should produce clear run failure.
        raise TrainingError(f"required synthetic training artifact is invalid: outputs/final_metrics.json: {exc}") from exc
    artifacts = final_metrics.get("artifacts") if isinstance(final_metrics, dict) else None
    if isinstance(artifacts, dict):
        prediction_samples = artifacts.get("prediction_samples")
        if isinstance(prediction_samples, str) and not (run_dir / prediction_samples).exists():
            raise TrainingError(f"required synthetic training artifact is missing: {prediction_samples}")
        best_metrics = artifacts.get("best_metrics")
        if isinstance(best_metrics, str) and not (run_dir / best_metrics).exists():
            raise TrainingError(f"required synthetic training artifact is missing: {best_metrics}")
        best_epoch_model = artifacts.get("best_epoch_model")
        if isinstance(best_epoch_model, str) and not (run_dir / best_epoch_model).exists():
            raise TrainingError(f"required synthetic training artifact is missing: {best_epoch_model}")
        return artifacts
    return None


def _artifacts_from_training_result(training_result: object) -> dict[str, object] | None:
    if isinstance(training_result, dict) and isinstance(training_result.get("artifacts"), dict):
        return training_result["artifacts"]
    return None


def _training_lifecycle_from_result(training_result: object) -> dict[str, object] | None:
    status = getattr(training_result, "lifecycle_status", None)
    timeout = getattr(training_result, "timeout", None)
    if status and status != "completed":
        lifecycle: dict[str, object] = {"status": str(status)}
        if isinstance(timeout, dict):
            lifecycle["timeout"] = timeout
        return lifecycle
    return None


def _merge_training_lifecycle(training_result: object, resource_lifecycle: dict[str, object]) -> dict[str, object]:
    lifecycle = dict(resource_lifecycle)
    result_lifecycle = _training_lifecycle_from_result(training_result)
    if result_lifecycle is None:
        return lifecycle
    lifecycle.update({key: value for key, value in result_lifecycle.items() if key != "status"})
    if result_lifecycle.get("status") != "completed":
        lifecycle["status"] = result_lifecycle["status"]
    return lifecycle


def _write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _package_version() -> str | None:
    try:
        return version("ml-autoresearch")
    except PackageNotFoundError:
        return None
