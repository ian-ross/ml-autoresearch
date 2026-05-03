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
from ml_autoresearch.execution import ExecutionBackend, NativeBackend, backend_metadata
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


@dataclass(frozen=True)
class RunSubmission:
    run_id: str
    run_dir: Path
    status: RunStatus
    rejection_reason: str | None = None


def run_candidate_with_synthetic_fixture(
    candidate_dir: str | Path,
    runs_root: str | Path,
    *,
    max_prediction_samples: int = 2,
    backend: ExecutionBackend | None = None,
) -> RunSubmission:
    """Validate, smoke-test, and synchronously train a Candidate Experiment Run."""

    return _run_candidate_synthetic_training(
        candidate_dir,
        runs_root,
        max_prediction_samples=max_prediction_samples,
        backend=backend,
    )


def run_candidate_with_gvccs_data(
    candidate_dir: str | Path,
    runs_root: str | Path,
    data_root: str | Path,
    *,
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    backend: ExecutionBackend | None = None,
) -> RunSubmission:
    """Validate, smoke-test, and synchronously train a Candidate Experiment Run on local GVCCS data."""

    data_path = _validate_host_data_root(data_root)
    selected_backend = backend or NativeBackend()
    return _run_candidate_training(
        candidate_dir,
        runs_root,
        lambda run_dir: selected_backend.train_gvccs(
            run_dir, data_path, max_samples=max_samples, max_prediction_samples=max_prediction_samples
        ),
        backend=selected_backend,
        dataset=_gvccs_dataset_metadata(data_path),
    )


def _run_candidate_synthetic_training(
    candidate_dir: str | Path,
    runs_root: str | Path,
    *,
    max_prediction_samples: int,
    backend: ExecutionBackend | None = None,
) -> RunSubmission:
    run = submit_candidate(candidate_dir, runs_root, backend=backend)
    if run.status != RunStatus.ACCEPTED:
        return run

    metadata = _read_metadata(run.run_dir)
    created_at = str(metadata["created_at"])
    candidate_source = Path(str(metadata["candidate_source"]["path"]))
    execution_backend_metadata = metadata.get("execution_backend")
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
    )
    selected_backend = backend or NativeBackend()
    try:
        selected_backend.train_synthetic(run.run_dir, max_prediction_samples=max_prediction_samples)
        artifacts = _validate_synthetic_training_outputs(run.run_dir)
    except (TrainingError, RuntimeError) as exc:
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
            execution_backend=execution_backend_metadata,
        )
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason)

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
    )
    return RunSubmission(run.run_id, run.run_dir, RunStatus.COMPLETED)


def _run_candidate_training(
    candidate_dir: str | Path,
    runs_root: str | Path,
    trainer,
    *,
    backend: ExecutionBackend | None = None,
    dataset: dict[str, object] | None = None,
) -> RunSubmission:
    run = submit_candidate(candidate_dir, runs_root, backend=backend)
    if run.status != RunStatus.ACCEPTED:
        return run

    metadata = _read_metadata(run.run_dir)
    created_at = str(metadata["created_at"])
    candidate_source = Path(str(metadata["candidate_source"]["path"]))
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
    )
    try:
        training_result = trainer(run.run_dir)
    except (TrainingError, RuntimeError, GVCCSDataError) as exc:
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
            execution_backend=metadata.get("execution_backend"),
            dataset=dataset,
        )
        return RunSubmission(run.run_id, run.run_dir, RunStatus.FAILED, reason)

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
    )
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
    }
    reason = metadata.get("rejection_reason") or metadata.get("smoke_failure_reason") or metadata.get("training_failure_reason")
    if reason is not None:
        summary["reason"] = reason
    if "artifacts" in metadata:
        summary["artifacts"] = metadata["artifacts"]

    final_metrics_path = _outputs_dir(run_dir) / "final_metrics.json"
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


def submit_candidate(candidate_dir: str | Path, runs_root: str | Path, *, backend: ExecutionBackend | None = None) -> RunSubmission:
    """Submit a local Candidate Experiment directory and create a Run record.

    Validation and smoke failures are represented as rejected/smoke_failed Runs
    so humans and agents can inspect logs and metadata for repair feedback.
    """

    source = Path(candidate_dir)
    execution_backend = backend or NativeBackend()
    execution_backend_metadata = backend_metadata(execution_backend)
    root = Path(runs_root)
    run_id = _generate_run_id(root)
    run_dir = root / run_id
    logs_dir = _outputs_dir(run_dir) / "logs"
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
            training_failure_reason=None,
            execution_backend=execution_backend_metadata,
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
        training_failure_reason=None,
        execution_backend=execution_backend_metadata,
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
            execution_backend=execution_backend_metadata,
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
        training_failure_reason=None,
        execution_backend=execution_backend_metadata,
    )
    return RunSubmission(run_id, run_dir, RunStatus.ACCEPTED)


def _outputs_dir(run_dir: Path) -> Path:
    return run_dir / "outputs"


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
    artifacts: dict[str, object] | None = None,
    execution_backend: object | None = None,
    dataset: dict[str, object] | None = None,
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
        "training_failure_reason": training_failure_reason,
    }
    if execution_backend is not None:
        metadata["execution_backend"] = execution_backend
    if dataset is not None:
        metadata["dataset"] = dataset
    if artifacts is not None:
        metadata["artifacts"] = artifacts
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
        return artifacts
    return None


def _artifacts_from_training_result(training_result: object) -> dict[str, object] | None:
    if isinstance(training_result, dict) and isinstance(training_result.get("artifacts"), dict):
        return training_result["artifacts"]
    return None


def _write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _package_version() -> str | None:
    try:
        return version("ml-autoresearch")
    except PackageNotFoundError:
        return None
