"""Run lifecycle for local Candidate Experiment submissions."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.errors import HarnessBootstrapError, ResearchProblemDataError, SmokeTestError, TrainingError
from ml_autoresearch.execution import DockerOperationTimeoutError, ExecutionBackend, NativeBackend, backend_metadata
from ml_autoresearch.managed_execution import cleanup_recorded_containers, read_execution_record
from ml_autoresearch.parameter_budget import DEFAULT_MAX_PARAMETER_COUNT
from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, ResearchLedgerError, record_research_event
from ml_autoresearch.research_problems import (
    ResearchProblemProviderConfig,
    ResearchProblemProviderLoadError,
    ResearchProblemSpecRegistry,
    legacy_smoke_research_problem_registry,
    load_research_problem_provider,
)


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


def _smoke_failure_classification(exc: BaseException) -> RunFailureClassification:
    """Distinguish trusted bootstrap faults from Candidate smoke failures."""

    current: BaseException | None = exc
    while current is not None:
        hint = getattr(current, "failure_classification", None)
        if hint == RunFailureClassification.HARNESS_FAILURE.value or isinstance(
            current,
            (HarnessBootstrapError, ResearchProblemProviderLoadError, ResearchProblemDataError),
        ):
            return RunFailureClassification.HARNESS_FAILURE
        current = current.__cause__
    return RunFailureClassification.CANDIDATE_BUG


def _training_failure_classification(
    exc: BaseException,
    *,
    run_dir: Path | None = None,
) -> RunFailureClassification:
    """Classify trusted training exceptions without letting Candidate code choose policy."""

    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, (HarnessBootstrapError, ResearchProblemProviderLoadError, ResearchProblemDataError)):
            return RunFailureClassification.HARNESS_FAILURE
        current_hint = getattr(current, "failure_classification", None)
        if current_hint == RunFailureClassification.HARNESS_FAILURE.value:
            return RunFailureClassification.HARNESS_FAILURE
        current = current.__cause__
    hint = getattr(exc, "failure_classification", None)
    if hint is None and run_dir is not None and "non_finite_training_state" in str(exc):
        diagnostic_path = run_dir / "outputs" / "nonfinite_diagnostic.json"
        if diagnostic_path.is_file():
            try:
                diagnostic = json.loads(diagnostic_path.read_text())
            except json.JSONDecodeError:
                diagnostic = {}
            if isinstance(diagnostic, dict):
                hint = diagnostic.get("failure_classification")
    if hint in {RunFailureClassification.CANDIDATE_BUG.value, RunFailureClassification.HARNESS_FAILURE.value}:
        return RunFailureClassification(hint)
    if is_resource_failure(exc):
        return RunFailureClassification.RESOURCE_FAILURE
    return RunFailureClassification.CANDIDATE_BUG


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
    max_parameters: int = DEFAULT_MAX_PARAMETER_COUNT,
    max_epochs: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
    provider_config: ResearchProblemProviderConfig | None = None,
) -> RunSubmission:
    """Compatibility wrapper that dispatches through a Research Problem provider."""

    if provider_config is None:
        raise CandidateValidationError("research_problem provider config is required for fixture Runs")
    return run_candidate_with_research_problem(
        candidate_dir,
        runs_root,
        provider_config,
        max_parameters=max_parameters,
        max_epochs=max_epochs,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
        backend=backend,
        ledger_path=ledger_path,
        require_proposal=require_proposal,
    )


def run_candidate_with_research_problem(
    candidate_dir: str | Path,
    runs_root: str | Path,
    provider_config: ResearchProblemProviderConfig,
    *,
    max_samples: int | None = None,
    max_parameters: int = DEFAULT_MAX_PARAMETER_COUNT,
    max_epochs: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
) -> RunSubmission:
    """Validate, smoke-test, and train through a generic Research Problem provider."""

    try:
        provider_config = provider_config.model_copy(update={"data_roots": provider_config.resolved_data_roots()})
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    loaded = load_research_problem_provider(provider_config)
    if not loaded.spec.operation_capabilities.training:
        raise TrainingError(f"Research Problem {loaded.spec.id!r} does not declare the training operation capability")
    if loaded.spec.training_adapter is None:
        raise TrainingError(f"Research Problem {loaded.spec.id!r} does not provide a training adapter")
    effective_data_config = provider_config.effective_data_config()
    validate_data_root = getattr(loaded.spec.training_adapter, "validate_data_root", None)
    if callable(validate_data_root):
        validate_data_root(effective_data_config)
    registry = ResearchProblemSpecRegistry(active_id=loaded.spec.id)
    registry.register(loaded.spec, provenance=loaded.provenance)
    selected_backend = backend or NativeBackend()
    return _run_candidate_training(
        candidate_dir,
        runs_root,
        lambda run_dir: selected_backend.train_research_problem(
            run_dir,
            provider_config,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        ),
        backend=selected_backend,
        dataset=loaded.spec.training_adapter.dataset_metadata(effective_data_config),
        research_problem=_research_problem_run_metadata(loaded.run_metadata(), provider_config),
        research_problem_registry=registry,
        ledger_path=ledger_path,
        require_proposal=require_proposal,
        max_parameters=max_parameters,
        max_epochs=max_epochs,
    )


def train_accepted_run_with_research_problem(
    run_dir: str | Path,
    provider_config: ResearchProblemProviderConfig,
    *,
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
) -> RunSubmission:
    """Synchronously train an accepted Run through a generic Research Problem provider."""

    path = Path(run_dir)
    metadata = _read_metadata(path)
    if metadata.get("status") != RunStatus.ACCEPTED.value:
        raise ValueError(f"accepted Run required for training continuation: {path}")
    try:
        provider_config = provider_config.model_copy(update={"data_roots": provider_config.resolved_data_roots()})
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc
    loaded = load_research_problem_provider(provider_config)
    if not loaded.spec.operation_capabilities.training:
        raise TrainingError(f"Research Problem {loaded.spec.id!r} does not declare the training operation capability")
    if loaded.spec.training_adapter is None:
        raise TrainingError(f"Research Problem {loaded.spec.id!r} does not provide a training adapter")
    selected_backend = backend or NativeBackend()
    effective_data_config = provider_config.effective_data_config()
    return _train_accepted_run(
        RunSubmission(str(metadata.get("run_id") or path.name), path, RunStatus.ACCEPTED),
        lambda accepted_run_dir: selected_backend.train_research_problem(
            accepted_run_dir,
            provider_config,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        ),
        backend=selected_backend,
        dataset=loaded.spec.training_adapter.dataset_metadata(effective_data_config),
        research_problem=_research_problem_run_metadata(loaded.run_metadata(), provider_config),
        ledger_path=_resolve_ledger_path(path.parent, ledger_path),
    )


def _research_problem_run_metadata(
    metadata: dict[str, object], provider_config: ResearchProblemProviderConfig
) -> dict[str, object]:
    if not provider_config.data_roots:
        return metadata
    resolved = dict(metadata)
    resolved["data_config"] = provider_config.model_dump(mode="json")["data_config"]
    resolved["data_roots"] = {
        name: {
            "host_path": str(path),
            "container_path": f"/data/{name}",
            "readonly": True,
        }
        for name, path in sorted(provider_config.data_roots.items())
    }
    return resolved


def train_accepted_run_with_synthetic_fixture(
    run_dir: str | Path,
    *,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
) -> RunSubmission:
    """Synchronously train an already accepted Candidate Experiment Run on synthetic fixture data."""

    path = Path(run_dir)
    metadata = _read_metadata(path)
    if metadata.get("status") != RunStatus.ACCEPTED.value:
        raise ValueError(f"accepted Run required for training continuation: {path}")
    selected_backend = backend or NativeBackend()
    return _train_accepted_run(
        RunSubmission(str(metadata.get("run_id") or path.name), path, RunStatus.ACCEPTED),
        lambda accepted_run_dir: selected_backend.train_synthetic(
            accepted_run_dir,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        ),
        backend=selected_backend,
        ledger_path=_resolve_ledger_path(path.parent, ledger_path),
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
    provider_config: ResearchProblemProviderConfig | None = None,
) -> RunSubmission:
    if provider_config is None:
        raise CandidateValidationError("research_problem provider config is required for synthetic fixture Runs")
    loaded = load_research_problem_provider(provider_config)
    registry = ResearchProblemSpecRegistry(active_id=loaded.spec.id)
    registry.register(loaded.spec, provenance=loaded.provenance)
    resolved_ledger_path = _resolve_ledger_path(runs_root, ledger_path)
    run = submit_candidate(
        candidate_dir,
        runs_root,
        backend=backend,
        ledger_path=resolved_ledger_path,
        require_proposal=require_proposal,
        research_problem_registry=registry,
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
        _cleanup_execution_containers(run.run_dir)
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason, RunFailureClassification.RESOURCE_FAILURE)
    except (TrainingError, RuntimeError) as exc:
        reason = str(exc)
        classification = _training_failure_classification(exc, run_dir=run.run_dir)
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
        _cleanup_execution_containers(run.run_dir)
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
        data_policy=_data_policy_from_training_result(training_result, run.run_dir),
        sample_counts=_sample_counts_from_training_result(training_result, run.run_dir),
        training_policy=_training_policy_from_training_result(training_result, run.run_dir),
    )
    _record_run_completed(resolved_ledger_path, run.run_id, run.run_dir)
    _cleanup_execution_containers(run.run_dir)
    return RunSubmission(run.run_id, run.run_dir, RunStatus.COMPLETED)


def _run_candidate_training(
    candidate_dir: str | Path,
    runs_root: str | Path,
    trainer,
    *,
    backend: ExecutionBackend | None = None,
    dataset: dict[str, object] | None = None,
    research_problem: dict[str, object] | None = None,
    research_problem_registry: ResearchProblemSpecRegistry | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
    max_parameters: int = DEFAULT_MAX_PARAMETER_COUNT,
    max_epochs: int | None = None,
) -> RunSubmission:
    resolved_ledger_path = _resolve_ledger_path(runs_root, ledger_path)
    run = submit_candidate(
        candidate_dir,
        runs_root,
        backend=backend,
        ledger_path=resolved_ledger_path,
        require_proposal=require_proposal,
        research_problem_registry=research_problem_registry,
        max_parameters=max_parameters,
        max_epochs=max_epochs,
    )
    if run.status != RunStatus.ACCEPTED:
        return run
    return _train_accepted_run(
        run,
        trainer,
        backend=backend,
        dataset=dataset,
        research_problem=research_problem,
        ledger_path=resolved_ledger_path,
    )


def _train_accepted_run(
    run: RunSubmission,
    trainer,
    *,
    backend: ExecutionBackend | None = None,
    dataset: dict[str, object] | None = None,
    research_problem: dict[str, object] | None = None,
    ledger_path: str | Path,
) -> RunSubmission:
    metadata = _read_metadata(run.run_dir)
    created_at = str(metadata["created_at"])
    candidate_source = Path(str(metadata["candidate_source"]["path"]))
    candidate_id = _candidate_id_from_run_dir(run.run_dir)
    record_research_event(
        "run_started",
        {"run_id": run.run_id, "candidate_id": candidate_id},
        ledger_path=ledger_path,
    )
    repair_lineage = metadata.get("repair_lineage") if isinstance(metadata.get("repair_lineage"), dict) else None
    execution_backend = metadata.get("execution_backend")
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
        execution_backend=execution_backend,
        dataset=dataset,
        research_problem=research_problem,
        repair_lineage=repair_lineage,
    )
    resource_lifecycle = None
    try:
        training_result, resource_lifecycle = _run_with_resource_retries(run.run_dir, lambda: trainer(run.run_dir))
        artifacts = _validate_training_outputs(run.run_dir, training_result)
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
            execution_backend=execution_backend,
            dataset=dataset,
            research_problem=research_problem,
            training_lifecycle={"status": "timeout_forced", "timeout": exc.timeout_metadata},
            repair_lineage=repair_lineage,
        )
        _record_run_failed(ledger_path, run.run_id, reason, RunFailureClassification.RESOURCE_FAILURE)
        _cleanup_execution_containers(run.run_dir)
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason, RunFailureClassification.RESOURCE_FAILURE)
    except (TrainingError, RuntimeError, ResearchProblemDataError) as exc:
        reason = str(exc)
        classification = _training_failure_classification(exc, run_dir=run.run_dir)
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
            execution_backend=execution_backend,
            dataset=dataset,
            research_problem=research_problem,
            training_lifecycle=failure_lifecycle,
            repair_lineage=repair_lineage,
        )
        _record_run_failed(ledger_path, run.run_id, reason, classification)
        _cleanup_execution_containers(run.run_dir)
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
        execution_backend=execution_backend,
        dataset=dataset,
        research_problem=research_problem,
        training_lifecycle=_merge_training_lifecycle(training_result, resource_lifecycle or {}),
        repair_lineage=repair_lineage,
        data_policy=_data_policy_from_training_result(training_result, run.run_dir),
        sample_counts=_sample_counts_from_training_result(training_result, run.run_dir),
        training_policy=_training_policy_from_training_result(training_result, run.run_dir),
    )
    _record_run_completed(ledger_path, run.run_id, run.run_dir)
    _cleanup_execution_containers(run.run_dir)
    return RunSubmission(run.run_id, run.run_dir, RunStatus.COMPLETED)


def fail_run_as_harness_failure(
    run_dir: str | Path,
    reason: str,
    *,
    ledger_path: str | Path | None = None,
) -> RunSubmission:
    """Idempotently terminalize an existing non-terminal Run after trusted orchestration failure."""

    path = Path(run_dir)
    resolved_ledger = _resolve_ledger_path(path.parent, ledger_path)
    with _run_finalization_lock(path):
        metadata = _read_metadata(path)
        run_id = str(metadata.get("run_id") or path.name)
        status = str(metadata.get("status"))
        if status == RunStatus.FAILED.value:
            return RunSubmission(
                run_id,
                path,
                RunStatus.FAILED,
                str(metadata.get("training_failure_reason") or reason),
                validate_run_failure_classification(metadata.get("failure_classification"))
                or RunFailureClassification.HARNESS_FAILURE,
            )
        if status in {RunStatus.COMPLETED.value, RunStatus.REJECTED.value, RunStatus.SMOKE_FAILED.value}:
            return RunSubmission(
                run_id,
                path,
                RunStatus(status),
                str(metadata.get("rejection_reason") or metadata.get("smoke_failure_reason") or "") or None,
                validate_run_failure_classification(metadata.get("failure_classification")),
            )
        classification = RunFailureClassification.HARNESS_FAILURE
        _update_terminal_metadata(
            path,
            metadata,
            status=RunStatus.FAILED,
            reason=reason,
            failure_classification=classification,
        )
        _ensure_terminal_event_once(
            resolved_ledger,
            run_id=run_id,
            event_type="run_failed",
            reason=reason,
            failure_classification=classification,
            run_dir=path,
        )
        _cleanup_execution_containers(path)
        return RunSubmission(run_id, path, RunStatus.FAILED, reason, classification)


def reconcile_run(
    run_dir: str | Path,
    *,
    ledger_path: str | Path | None = None,
) -> RunSubmission:
    """Idempotently validate and terminalize one existing Run without retraining."""

    path = Path(run_dir)
    resolved_ledger = _resolve_ledger_path(path.parent, ledger_path)
    with _run_finalization_lock(path):
        metadata = _read_metadata(path)
        run_id = str(metadata.get("run_id") or path.name)
        status = str(metadata.get("status"))
        if status == RunStatus.SMOKE_TESTING.value:
            execution = read_execution_record(path)
            if isinstance(execution, dict) and (
                execution.get("supervisor_alive") is True
                or execution.get("observed_state") == "container_running"
            ):
                return RunSubmission(run_id, path, RunStatus.SMOKE_TESTING)
            reason = "managed pre-training smoke execution is no longer active"
            if isinstance(execution, dict) and execution.get("error"):
                reason = f"{reason}: {execution['error']}"
            classification = RunFailureClassification.HARNESS_FAILURE
            _update_terminal_metadata(
                path,
                metadata,
                status=RunStatus.FAILED,
                reason=reason,
                failure_classification=classification,
            )
            _ensure_terminal_event_once(
                resolved_ledger,
                run_id=run_id,
                event_type="run_failed",
                reason=reason,
                failure_classification=classification,
                run_dir=path,
            )
            _cleanup_execution_containers(path)
            return RunSubmission(run_id, path, RunStatus.FAILED, reason, classification)
        if status in {RunStatus.REJECTED.value, RunStatus.SMOKE_FAILED.value}:
            return RunSubmission(
                run_id,
                path,
                RunStatus(status),
                str(metadata.get("rejection_reason") or metadata.get("smoke_failure_reason") or "") or None,
                validate_run_failure_classification(metadata.get("failure_classification")),
            )
        if status == RunStatus.FAILED.value:
            classification = (
                validate_run_failure_classification(metadata.get("failure_classification"))
                or RunFailureClassification.UNKNOWN
            )
            reason = str(metadata.get("training_failure_reason") or "unknown failure")
            _ensure_terminal_event_once(
                resolved_ledger,
                run_id=run_id,
                event_type="run_failed",
                reason=reason,
                failure_classification=classification,
                run_dir=path,
            )
            _cleanup_execution_containers(path)
            return RunSubmission(run_id, path, RunStatus.FAILED, reason, classification)
        if status == RunStatus.ACCEPTED.value:
            return RunSubmission(run_id, path, RunStatus.ACCEPTED)
        if status == RunStatus.TRAINING.value:
            execution = read_execution_record(path)
            if isinstance(execution, dict) and (
                execution.get("supervisor_alive") is True
                or execution.get("observed_state") == "container_running"
            ):
                return RunSubmission(run_id, path, RunStatus.TRAINING)
            if isinstance(execution, dict):
                container_observation = execution.get("container_observation")
                active_container = execution.get("active_container")
                exit_code = (
                    container_observation.get("exit_code")
                    if isinstance(container_observation, dict) and container_observation.get("status") == "exited"
                    else active_container.get("exit_code")
                    if isinstance(active_container, dict) and active_container.get("state") == "exited_failure"
                    else None
                )
                if isinstance(exit_code, int) and exit_code != 0:
                    container_error = (
                        container_observation.get("error")
                        if isinstance(container_observation, dict)
                        else active_container.get("error") if isinstance(active_container, dict) else None
                    )
                    reason = (
                        f"managed Docker operation exited with status {exit_code}: "
                        f"{container_error or 'no container error detail'}"
                    )
                    oom_killed = bool(
                        isinstance(container_observation, dict) and container_observation.get("oom_killed") is True
                    )
                    classification = (
                        RunFailureClassification.RESOURCE_FAILURE
                        if oom_killed or is_resource_failure(reason)
                        else RunFailureClassification.CANDIDATE_BUG
                    )
                    _update_terminal_metadata(
                        path,
                        metadata,
                        status=RunStatus.FAILED,
                        reason=reason,
                        failure_classification=classification,
                    )
                    _ensure_terminal_event_once(
                        resolved_ledger,
                        run_id=run_id,
                        event_type="run_failed",
                        reason=reason,
                        failure_classification=classification,
                        run_dir=path,
                    )
                    _cleanup_execution_containers(path)
                    return RunSubmission(run_id, path, RunStatus.FAILED, reason, classification)
        try:
            artifacts = _validate_training_outputs(path, None)
        except (TrainingError, RuntimeError) as exc:
            reason = f"Run reconciliation artifact validation failed: {exc}"
            classification = _training_failure_classification(exc, run_dir=path)
            if (
                classification == RunFailureClassification.CANDIDATE_BUG
                and not str(exc).startswith("non_finite_training_state:")
            ):
                classification = RunFailureClassification.HARNESS_FAILURE
            _update_terminal_metadata(
                path,
                metadata,
                status=RunStatus.FAILED,
                reason=reason,
                failure_classification=classification,
            )
            _ensure_terminal_event_once(
                resolved_ledger,
                run_id=run_id,
                event_type="run_failed",
                reason=reason,
                failure_classification=classification,
                run_dir=path,
            )
            _cleanup_execution_containers(path)
            return RunSubmission(run_id, path, RunStatus.FAILED, reason, classification)
        _update_terminal_metadata(
            path,
            metadata,
            status=RunStatus.COMPLETED,
            artifacts=artifacts,
        )
        _ensure_terminal_event_once(
            resolved_ledger,
            run_id=run_id,
            event_type="run_completed",
            run_dir=path,
        )
        _cleanup_execution_containers(path)
        return RunSubmission(run_id, path, RunStatus.COMPLETED)


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
    if isinstance(metadata.get("research_problem"), dict):
        summary["research_problem"] = metadata["research_problem"]
    if "artifacts" in metadata:
        summary["artifacts"] = metadata["artifacts"]
    if isinstance(metadata.get("training_policy"), dict):
        summary["training_policy"] = metadata["training_policy"]

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


def _research_problem_identity(
    manifest: object,
    registry: ResearchProblemSpecRegistry | None = None,
) -> dict[str, object]:
    spec_id = str(getattr(manifest, "research_problem"))
    if registry is None:
        raise CandidateValidationError("research_problem provider registry is required")
    spec = registry.get(spec_id)
    identity: dict[str, object] = {"id": spec.id, "version": spec.version, "contract_version": spec.contract_version}
    provenance = registry.get_provenance(spec_id)
    if provenance is not None:
        identity["provider"] = provenance.run_metadata()
    return identity


def _resolved_manifest_payload(
    manifest: object,
    registry: ResearchProblemSpecRegistry | None = None,
) -> dict[str, object]:
    payload = manifest.model_dump(mode="json")
    payload["research_problem"] = _research_problem_identity(manifest, registry)
    data_policy = payload.setdefault("data", {})
    if data_policy.get("temporal_offsets_seconds") is None:
        data_policy.pop("temporal_offsets_seconds", None)
    selected = data_policy.get("augmentation_policy", "none")
    data_policy["augmentation_policy"] = selected
    data_policy["augmentation_policy_effective"] = selected
    frame_selection = data_policy.get("frame_selection_policy", "all_target_frames")
    data_policy["frame_selection_policy"] = frame_selection
    data_policy["frame_selection_policy_effective"] = frame_selection
    if payload.get("input_mode") == "centered_temporal_rgb_clip":
        temporal_offsets = data_policy.get("temporal_offsets_seconds", [-30, 0, 30])
        data_policy["temporal_offsets_seconds"] = temporal_offsets
        data_policy["temporal_offsets_seconds_effective"] = temporal_offsets
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
    with ledger_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResearchLedgerError(
                    f"malformed JSON in ledger {ledger_path} at line {line_number}: cannot enforce repair limits safely"
                ) from exc
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



def prepare_candidate_submission(
    candidate_dir: str | Path,
    runs_root: str | Path,
    *,
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
    research_problem_registry: ResearchProblemSpecRegistry | None = None,
    max_epochs: int | None = None,
) -> RunSubmission:
    """Validate and copy a Candidate into a stable Run before smoke execution."""

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

    if research_problem_registry is None:
        research_problem_registry = legacy_smoke_research_problem_registry()

    try:
        manifest = validate_candidate_directory(
            source,
            require_proposal=require_proposal,
            research_problem_registry=research_problem_registry,
            max_epochs=max_epochs,
        )
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

    research_problem = _research_problem_identity(manifest, research_problem_registry)
    validation_log.write_text("Candidate validation accepted.\n")
    proposal_path = source / "PROPOSAL.md"
    if proposal_path.is_file():
        _record_proposal_created_event(proposal_path, manifest.name, resolved_ledger_path=resolved_ledger_path)
    _record_candidate_created_event(source, manifest.name, proposal_id=manifest.name if proposal_path.is_file() else None, repair_lineage=repair_lineage, resolved_ledger_path=resolved_ledger_path)
    shutil.copytree(source, run_dir / "candidate")
    _write_yaml(run_dir / "resolved_manifest.yaml", _resolved_manifest_payload(manifest, research_problem_registry))
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
        research_problem=research_problem,
    )

    return RunSubmission(run_id, run_dir, RunStatus.SMOKE_TESTING)


def smoke_test_prepared_run(
    run_dir: str | Path,
    *,
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    max_parameters: int = DEFAULT_MAX_PARAMETER_COUNT,
) -> RunSubmission:
    """Smoke-test one stable prepared Run without creating another Run."""

    path = Path(run_dir)
    metadata = _read_metadata(path)
    run_id = str(metadata.get("run_id") or path.name)
    status = RunStatus(str(metadata.get("status")))
    if status in {RunStatus.ACCEPTED, RunStatus.SMOKE_FAILED}:
        return RunSubmission(
            run_id,
            path,
            status,
            str(metadata.get("smoke_failure_reason") or "") or None,
            validate_run_failure_classification(metadata.get("failure_classification")),
        )
    if status != RunStatus.SMOKE_TESTING:
        raise ValueError(f"smoke-testing Run required for continuation: {path}")

    execution_backend = backend or NativeBackend()
    execution_backend_metadata = metadata.get("execution_backend") or backend_metadata(execution_backend)
    source = Path(str(metadata["candidate_source"]["path"]))
    created_at = str(metadata["created_at"])
    repair_lineage = metadata.get("repair_lineage") if isinstance(metadata.get("repair_lineage"), dict) else None
    research_problem = metadata.get("research_problem") if isinstance(metadata.get("research_problem"), dict) else None
    resolved_ledger_path = _resolve_ledger_path(path.parent, ledger_path)

    try:
        try:
            execution_backend.smoke_test(path, max_parameters=max_parameters)
        except TypeError as exc:
            # Preserve compatibility for test/developer backends implementing the
            # pre-budget protocol. Non-default budgets require the trusted API.
            if max_parameters != DEFAULT_MAX_PARAMETER_COUNT or "max_parameters" not in str(exc):
                raise
            execution_backend.smoke_test(path)
    except (SmokeTestError, RuntimeError) as exc:
        reason = str(exc)
        classification = _smoke_failure_classification(exc)
        _write_metadata(
            path,
            run_id=run_id,
            status=RunStatus.SMOKE_FAILED,
            created_at=created_at,
            updated_at=_now_iso(),
            candidate_source=source,
            rejection_reason=None,
            smoke_failure_reason=reason,
            training_failure_reason=None,
            failure_classification=classification,
            execution_backend=execution_backend_metadata,
            repair_lineage=repair_lineage,
            research_problem=research_problem,
        )
        return RunSubmission(run_id, path, RunStatus.SMOKE_FAILED, reason, classification)

    _write_metadata(
        path,
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
        research_problem=research_problem,
    )
    record_research_event(
        "candidate_submitted",
        {"candidate_id": _candidate_id_from_run_dir(path), "run_id": run_id},
        ledger_path=resolved_ledger_path,
    )
    return RunSubmission(run_id, path, RunStatus.ACCEPTED)


def submit_candidate(
    candidate_dir: str | Path,
    runs_root: str | Path,
    *,
    backend: ExecutionBackend | None = None,
    ledger_path: str | Path | None = None,
    require_proposal: bool = False,
    research_problem_registry: ResearchProblemSpecRegistry | None = None,
    max_parameters: int = DEFAULT_MAX_PARAMETER_COUNT,
    max_epochs: int | None = None,
) -> RunSubmission:
    """Validate and synchronously smoke-test a local Candidate Experiment."""

    prepared = prepare_candidate_submission(
        candidate_dir,
        runs_root,
        backend=backend,
        ledger_path=ledger_path,
        require_proposal=require_proposal,
        research_problem_registry=research_problem_registry,
        max_epochs=max_epochs,
    )
    if prepared.status != RunStatus.SMOKE_TESTING:
        return prepared
    return smoke_test_prepared_run(
        prepared.run_dir,
        backend=backend,
        ledger_path=ledger_path,
        max_parameters=max_parameters,
    )


@contextmanager
def _run_finalization_lock(run_dir: Path):
    """Serialize terminal metadata and Research Ledger transitions for one Run."""

    lock_path = run_dir / ".finalization.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _update_terminal_metadata(
    run_dir: Path,
    metadata: dict[str, object],
    *,
    status: RunStatus,
    reason: str | None = None,
    failure_classification: RunFailureClassification | None = None,
    artifacts: dict[str, object] | None = None,
) -> None:
    updated = dict(metadata)
    updated["status"] = status.value
    updated["updated_at"] = _now_iso()
    updated["training_failure_reason"] = reason
    updated["failure_classification"] = failure_classification.value if failure_classification is not None else None
    if artifacts is not None:
        updated["artifacts"] = artifacts
    temporary = run_dir / f".run_metadata.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, run_dir / "run_metadata.json")


def _ensure_terminal_event_once(
    ledger_path: Path,
    *,
    run_id: str,
    event_type: str,
    run_dir: Path,
    reason: str | None = None,
    failure_classification: RunFailureClassification = RunFailureClassification.UNKNOWN,
) -> None:
    terminal_events: list[dict[str, object]] = []
    if ledger_path.is_file():
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(event, dict)
                and event.get("run_id") == run_id
                and event.get("event_type") in {"run_completed", "run_failed"}
            ):
                terminal_events.append(event)
    if len(terminal_events) > 1:
        raise ResearchLedgerError(f"Run {run_id} has duplicate terminal Research Ledger events")
    if terminal_events:
        existing_type = terminal_events[0].get("event_type")
        if existing_type != event_type:
            raise ResearchLedgerError(
                f"Run {run_id} terminal state conflicts with existing {existing_type} Research Ledger event"
            )
        return
    if event_type == "run_completed":
        _record_run_completed(ledger_path, run_id, run_dir)
        return
    _record_run_failed(ledger_path, run_id, reason or "unknown failure", failure_classification)


def _cleanup_execution_containers(run_dir: Path) -> None:
    cleanup_recorded_containers(run_dir)


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


@contextmanager
def _ledger_terminal_lock(ledger_path: Path):
    lock_path = ledger_path.with_name(f".{ledger_path.name}.terminal.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _terminal_events_for_run(ledger_path: Path, run_id: str) -> list[dict[str, object]]:
    if not ledger_path.is_file():
        return []
    events: list[dict[str, object]] = []
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("run_id") == run_id
            and event.get("event_type") in {"run_completed", "run_failed"}
        ):
            events.append(event)
    return events


def _record_run_completed(ledger_path: Path, run_id: str, run_dir: Path) -> None:
    with _ledger_terminal_lock(ledger_path):
        terminal_events = _terminal_events_for_run(ledger_path, run_id)
        if len(terminal_events) > 1:
            raise ResearchLedgerError(f"Run {run_id} has duplicate terminal Research Ledger events")
        if terminal_events:
            if terminal_events[0].get("event_type") != "run_completed":
                raise ResearchLedgerError(f"Run {run_id} already has a conflicting run_failed event")
            return
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
    with _ledger_terminal_lock(ledger_path):
        terminal_events = _terminal_events_for_run(ledger_path, run_id)
        if len(terminal_events) > 1:
            raise ResearchLedgerError(f"Run {run_id} has duplicate terminal Research Ledger events")
        if terminal_events:
            if terminal_events[0].get("event_type") != "run_failed":
                raise ResearchLedgerError(f"Run {run_id} already has a conflicting run_completed event")
            return
        record_research_event(
            "run_failed",
            {
                "run_id": run_id,
                "error": reason or "unknown failure",
                "failure_classification": failure_classification.value,
            },
            ledger_path=ledger_path,
        )


def _data_policy_from_training_result(training_result: object, run_dir: Path) -> dict[str, object] | None:
    if isinstance(training_result, dict) and isinstance(training_result.get("data_policy"), dict):
        return training_result["data_policy"]
    final_metrics = _read_final_metrics_if_available(run_dir)
    if isinstance(final_metrics.get("data_policy"), dict):
        return final_metrics["data_policy"]
    return None


def _sample_counts_from_training_result(training_result: object, run_dir: Path) -> dict[str, object] | None:
    if isinstance(training_result, dict) and isinstance(training_result.get("sample_counts"), dict):
        return training_result["sample_counts"]
    final_metrics = _read_final_metrics_if_available(run_dir)
    if isinstance(final_metrics.get("sample_counts"), dict):
        return final_metrics["sample_counts"]
    return None


def _training_policy_from_training_result(training_result: object, run_dir: Path) -> dict[str, object] | None:
    if isinstance(training_result, dict) and isinstance(training_result.get("training_policy"), dict):
        return training_result["training_policy"]
    final_metrics = _read_final_metrics_if_available(run_dir)
    if isinstance(final_metrics.get("training_policy"), dict):
        return final_metrics["training_policy"]
    return None


def _read_final_metrics_if_available(run_dir: Path) -> dict[str, object]:
    final_metrics_path = run_dir / "outputs" / "final_metrics.json"
    if not final_metrics_path.is_file():
        return {}
    try:
        data = json.loads(final_metrics_path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


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
    research_problem: dict[str, object] | None = None,
    data_policy: dict[str, object] | None = None,
    sample_counts: dict[str, object] | None = None,
    training_policy: dict[str, object] | None = None,
) -> None:
    existing_metadata = None
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        try:
            existing_metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            existing_metadata = None
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
    if research_problem is not None:
        metadata["research_problem"] = research_problem
    elif isinstance(existing_metadata, dict) and isinstance(existing_metadata.get("research_problem"), dict):
        metadata["research_problem"] = existing_metadata["research_problem"]
    if execution_backend is not None:
        metadata["execution_backend"] = execution_backend
    if dataset is not None:
        metadata["dataset"] = dataset
    if artifacts is not None:
        metadata["artifacts"] = artifacts
    if data_policy is not None:
        metadata["data_policy"] = data_policy
    elif isinstance(existing_metadata, dict) and isinstance(existing_metadata.get("data_policy"), dict):
        metadata["data_policy"] = existing_metadata["data_policy"]
    if sample_counts is not None:
        metadata["sample_counts"] = sample_counts
    elif isinstance(existing_metadata, dict) and isinstance(existing_metadata.get("sample_counts"), dict):
        metadata["sample_counts"] = existing_metadata["sample_counts"]
    if training_policy is not None:
        metadata["training_policy"] = training_policy
    elif isinstance(existing_metadata, dict) and isinstance(existing_metadata.get("training_policy"), dict):
        metadata["training_policy"] = existing_metadata["training_policy"]
    if training_lifecycle is not None:
        metadata["training_lifecycle"] = training_lifecycle
    if repair_lineage is not None:
        metadata["repair_lineage"] = repair_lineage
    if isinstance(existing_metadata, dict):
        for batch_field in ("batch_id", "batch_candidate_id"):
            if batch_field in existing_metadata:
                metadata[batch_field] = existing_metadata[batch_field]
    temporary = run_dir / f".run_metadata.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, metadata_path)


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


def _validate_training_outputs(run_dir: Path, training_result: object) -> dict[str, object] | None:
    from ml_autoresearch.finite import require_finite_json_numbers, require_finite_tensor

    outputs_dir = _outputs_dir(run_dir)
    required = [
        outputs_dir / "metrics.jsonl",
        outputs_dir / "final_metrics.json",
        outputs_dir / "best_metrics.json",
        outputs_dir / "logs" / "training.log",
    ]
    for path in required:
        if not path.exists():
            raise TrainingError(
                f"required training artifact is missing: {path.relative_to(run_dir)}; "
                "required synthetic training artifact is missing"
            )
    try:
        metric_records = [
            json.loads(line)
            for line in (outputs_dir / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        final_metrics = json.loads((outputs_dir / "final_metrics.json").read_text())
        best_metrics = json.loads((outputs_dir / "best_metrics.json").read_text())
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"required training metric artifact is invalid: {exc}") from exc
    for index, record in enumerate(metric_records):
        require_finite_json_numbers(
            record,
            outputs_dir=outputs_dir,
            phase="terminal_validation",
            checkpoint="metrics_artifacts",
            quantity_prefix=f"metric.metrics.jsonl[{index}]",
        )
    require_finite_json_numbers(
        final_metrics,
        outputs_dir=outputs_dir,
        phase="terminal_validation",
        checkpoint="metrics_artifacts",
        quantity_prefix="metric.final_metrics",
    )
    require_finite_json_numbers(
        best_metrics,
        outputs_dir=outputs_dir,
        phase="terminal_validation",
        checkpoint="metrics_artifacts",
        quantity_prefix="metric.best_metrics",
    )
    artifacts = (
        training_result["artifacts"]
        if isinstance(training_result, dict) and isinstance(training_result.get("artifacts"), dict)
        else final_metrics.get("artifacts")
        if isinstance(final_metrics, dict) and isinstance(final_metrics.get("artifacts"), dict)
        else None
    )
    validation_reports_relative = artifacts.get("validation_postprocessing") if isinstance(artifacts, dict) else None
    if isinstance(validation_reports_relative, str):
        validation_epochs = [
            int(record["epoch"])
            for record in metric_records
            if isinstance(record, dict)
            and record.get("split") == "val"
            and isinstance(record.get("epoch"), int)
        ]
        _validate_validation_postprocessing_artifacts(
            run_dir,
            outputs_dir,
            validation_reports_relative,
            expected_epochs=validation_epochs,
            require_finite_json_numbers=require_finite_json_numbers,
        )
    checkpoint_relative = artifacts.get("best_epoch_model") if isinstance(artifacts, dict) else None
    if not isinstance(checkpoint_relative, str) and isinstance(best_metrics, dict):
        checkpoint_relative = best_metrics.get("model_artifact")
    if not isinstance(checkpoint_relative, str) or not checkpoint_relative:
        raise TrainingError("required training checkpoint reference is missing")
    relative_checkpoint = Path(checkpoint_relative)
    if relative_checkpoint.is_absolute() or ".." in relative_checkpoint.parts:
        raise TrainingError(f"required training checkpoint path is invalid: {checkpoint_relative}")
    checkpoint_path = run_dir / relative_checkpoint
    if not checkpoint_path.is_file():
        raise TrainingError(f"required training artifact is missing: {checkpoint_relative}")
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"required training checkpoint is invalid: {checkpoint_relative}: {exc}") from exc
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise TrainingError(f"required training checkpoint is invalid: {checkpoint_relative}: missing model_state_dict")
    for tensor_name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            require_finite_tensor(
                tensor,
                outputs_dir=outputs_dir,
                phase="terminal_validation",
                checkpoint="checkpoint_tensors",
                failing_quantity=f"checkpoint.model_state_dict.{tensor_name}",
            )
    return artifacts


def _validate_validation_postprocessing_artifacts(
    run_dir: Path,
    outputs_dir: Path,
    index_relative: str,
    *,
    expected_epochs: list[int],
    require_finite_json_numbers,
) -> None:
    relative_index = Path(index_relative)
    if relative_index.is_absolute() or ".." in relative_index.parts:
        raise TrainingError(f"validation postprocessing artifact path is invalid: {index_relative}")
    index_path = run_dir / relative_index
    if not index_path.is_file():
        raise TrainingError(f"required training artifact is missing: {index_relative}")
    try:
        index = json.loads(index_path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"validation postprocessing artifact is invalid: {index_relative}: {exc}") from exc
    if not isinstance(index, dict) or index.get("schema_version") != "validation_postprocessing_index.v1":
        raise TrainingError(f"validation postprocessing artifact is invalid: {index_relative}: unexpected schema")
    reports = index.get("reports")
    if not isinstance(reports, list) or not reports or not all(isinstance(item, str) for item in reports):
        raise TrainingError(f"validation postprocessing artifact is invalid: {index_relative}: reports must be non-empty")
    if len(reports) != len(set(reports)) or len(reports) != len(expected_epochs):
        raise TrainingError(
            f"validation postprocessing artifact is invalid: {index_relative}: "
            "reports must uniquely match completed validation epochs"
        )
    require_finite_json_numbers(
        index,
        outputs_dir=outputs_dir,
        phase="terminal_validation",
        checkpoint="validation_postprocessing_artifacts",
        quantity_prefix="validation.postprocessing.index",
        failure_classification="harness_failure",
    )
    observed_epochs: list[int] = []
    for report_number, report_relative in enumerate(reports):
        relative_report = Path(report_relative)
        if relative_report.is_absolute() or ".." in relative_report.parts:
            raise TrainingError(f"validation postprocessing artifact path is invalid: {report_relative}")
        report_path = run_dir / relative_report
        if not report_path.is_file():
            raise TrainingError(f"required training artifact is missing: {report_relative}")
        try:
            report = json.loads(report_path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise TrainingError(f"validation postprocessing artifact is invalid: {report_relative}: {exc}") from exc
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != "validation_postprocessing_epoch.v1"
            or report.get("status") != "completed"
        ):
            raise TrainingError(f"validation postprocessing artifact is invalid: {report_relative}: unexpected schema/status")
        epoch = report.get("epoch")
        inference = report.get("inference")
        postprocessing = report.get("postprocessing")
        if (
            not isinstance(epoch, int)
            or not isinstance(inference, dict)
            or not isinstance(inference.get("device"), str)
            or not isinstance(inference.get("sample_count"), int)
            or int(inference["sample_count"]) < 1
            or not isinstance(inference.get("elapsed_seconds"), (int, float))
            or not isinstance(postprocessing, dict)
            or postprocessing.get("backend") not in {"torch_cpu", "torch_cuda"}
            or not isinstance(postprocessing.get("requested_device"), str)
            or not isinstance(postprocessing.get("device"), str)
            or not isinstance(postprocessing.get("batch_size"), int)
            or int(postprocessing["batch_size"]) < 1
            or not isinstance(postprocessing.get("max_device_batch_samples"), int)
            or not 1 <= int(postprocessing["max_device_batch_samples"]) <= int(postprocessing["batch_size"])
            or postprocessing.get("bounded_device_batches") is not True
            or postprocessing.get("full_validation_gpu_residency") is not False
            or not isinstance(postprocessing.get("timings_seconds"), dict)
            or not postprocessing["timings_seconds"]
        ):
            raise TrainingError(
                f"validation postprocessing artifact is invalid: {report_relative}: "
                "required postprocessing evidence is missing or invalid"
            )
        observed_epochs.append(epoch)
        require_finite_json_numbers(
            report,
            outputs_dir=outputs_dir,
            phase="terminal_validation",
            checkpoint="validation_postprocessing_artifacts",
            quantity_prefix=f"validation.postprocessing.reports[{report_number}]",
            failure_classification="harness_failure",
        )
    if observed_epochs != expected_epochs:
        raise TrainingError(
            f"validation postprocessing artifact is invalid: {index_relative}: "
            "report epochs do not match completed validation epochs"
        )


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
