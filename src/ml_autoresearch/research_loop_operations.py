"""Reusable Research Loop operation modules shared by CLI and Autonomy Steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from ml_autoresearch.candidate_execution_config import (
    CONFIG_FILENAME,
    CandidateExecutionConfigError,
    execution_backend_from_config,
    load_candidate_execution_config,
    load_configured_research_problem_registry,
    resolve_configured_research_problem_provider,
)
from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, DockerBackend, ExecutionBackend, NativeBackend
from ml_autoresearch.runs import RunStatus

PredictionSamplePolicy = Literal["first_n", "adjacent_and_scattered"]


class ResearchLoopOperationError(RuntimeError):
    """Raised when a Research Loop operation cannot be executed."""


def select_execution_backend(
    name: str,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    *,
    docker_enable_gpu: bool = False,
    docker_user: str | None = None,
    docker_rootless_container_root: bool = False,
) -> ExecutionBackend:
    """Return a concrete execution adapter from CLI/config choices."""

    if name == "native":
        if docker_enable_gpu:
            raise ValueError("--docker-enable-gpu requires --backend docker")
        if docker_user is not None:
            raise ValueError("--docker-user requires --backend docker")
        if docker_rootless_container_root:
            raise ValueError("--docker-rootless-container-root requires --backend docker")
        return NativeBackend()
    if name == "docker":
        if docker_user is not None and docker_rootless_container_root:
            raise ValueError("choose either --docker-user or --docker-rootless-container-root, not both")
        return DockerBackend(
            docker_image,
            enable_gpu=docker_enable_gpu,
            container_user=docker_user,
            rootless_container_root=docker_rootless_container_root,
        )
    raise ValueError("backend must be native or docker")


def load_configured_provider(workspace_root: str | Path, *, label: str = "run"):
    """Load configured Research Problem provider for a Research Workspace Root."""

    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise CandidateExecutionConfigError(f"workspace-root does not exist or is not a directory: {root}")
    config_path = root / CONFIG_FILENAME
    if not config_path.is_file():
        raise CandidateExecutionConfigError(
            f"{config_path} not found; create a Research Problem execution config before {label}"
        )

    config = load_candidate_execution_config(root)
    provider_config = resolve_configured_research_problem_provider(config)
    if provider_config is None:
        raise CandidateExecutionConfigError(f"configure [research_problem] in {config_path}")

    resolved_data_root = resolve_research_problem_data_root(provider_config.data_config, workspace_root=root)
    if resolved_data_root is not None:
        provider_config = resolve_configured_research_problem_provider(config, data_root_override=resolved_data_root)
        if provider_config is None:
            raise CandidateExecutionConfigError(f"configure [research_problem] in {config_path}")
    return config, provider_config


def resolve_research_problem_data_root(configured_data_config: dict[str, object], *, workspace_root: str | Path) -> Path | None:
    """Resolve a configured Research Problem data root relative to the Research Workspace Root."""

    data_root = configured_data_config.get("dataset_root") or configured_data_config.get("data_root")
    if data_root is None:
        return None
    if not isinstance(data_root, str):
        raise CandidateExecutionConfigError("research_problem.data_config.dataset_root must be a string")
    candidate_root = Path(data_root)
    if not candidate_root.is_absolute():
        candidate_root = Path(workspace_root) / candidate_root
    if not candidate_root.exists():
        raise CandidateExecutionConfigError(f"Research Problem data root does not exist: {candidate_root}")
    if not candidate_root.is_dir():
        raise CandidateExecutionConfigError(f"Research Problem data root is not a directory: {candidate_root}")
    return candidate_root


def effective_execution_options(
    config,
    *,
    max_samples: int | None,
    max_prediction_samples: int | None,
    prediction_sample_policy: str | None,
) -> tuple[int | None, int, str]:
    """Merge per-command execution options with Workspace Configuration defaults."""

    return (
        config.max_samples if max_samples is None else max_samples,
        config.max_prediction_samples if max_prediction_samples is None else max_prediction_samples,
        config.prediction_sample_policy if prediction_sample_policy is None else prediction_sample_policy,
    )


def effective_ledger_path(config, *, override: Path | None) -> Path:
    """Resolve the Research Ledger path for a Harness-owned operation."""

    if override is not None:
        return override
    if config.ledger_path is None:
        raise CandidateExecutionConfigError(
            "configure candidate_execution.ledger_path in ml-autoresearch.toml or pass --ledger-path"
        )
    return config.ledger_path


def run_submission_payload(run) -> dict[str, object]:
    """Return the stable JSON payload for a Run submission/result."""

    return {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "status": run.status.value,
        "rejection_reason": run.rejection_reason,
        "failure_classification": run.failure_classification.value if run.failure_classification is not None else None,
    }


def run_candidate_from_workspace(
    candidate: str | Path,
    *,
    workspace_root: str | Path = Path("."),
    runs_root: str | Path | None = None,
    backend_name: str = "docker",
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    docker_enable_gpu: bool = False,
    docker_user: str | None = None,
    docker_rootless_container_root: bool = False,
    max_samples: int | None = None,
    max_prediction_samples: int | None = None,
    prediction_sample_policy: str | None = None,
    ledger_path: Path | None = None,
    require_proposal: bool = True,
) -> dict[str, object]:
    """Run one Candidate Experiment using Workspace Configuration and return a JSON payload."""

    from ml_autoresearch.runs import run_candidate_with_research_problem

    config, provider_config = load_configured_provider(workspace_root, label="run-candidate")
    effective_runs_root = config.runs_root if runs_root is None else Path(runs_root)
    effective_ledger = effective_ledger_path(config, override=ledger_path)
    backend = select_execution_backend(
        backend_name,
        docker_image,
        docker_enable_gpu=docker_enable_gpu,
        docker_user=docker_user,
        docker_rootless_container_root=docker_rootless_container_root,
    )
    max_samples, max_prediction_samples, prediction_sample_policy = effective_execution_options(
        config,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
    )
    run = run_candidate_with_research_problem(
        candidate,
        effective_runs_root,
        provider_config,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
        backend=backend,
        ledger_path=effective_ledger,
        require_proposal=require_proposal,
    )
    return run_submission_payload(run)


def run_experiment_batch_from_workspace(
    batch: str | Path,
    *,
    batches_root: str | Path,
    runs_root: str | Path,
    workspace_root: str | Path = Path("."),
    backend_name: str = "docker",
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    docker_enable_gpu: bool = False,
    docker_user: str | None = None,
    docker_rootless_container_root: bool = False,
    max_parallel_runs: int = 4,
    max_samples: int | None = None,
    max_prediction_samples: int | None = None,
    prediction_sample_policy: str | None = None,
    ledger_path: Path | None = None,
) -> dict[str, object]:
    """Run an Experiment Batch using Workspace Configuration and return its serializable result."""

    from ml_autoresearch.batches import run_experiment_batch_with_research_problem

    config, provider_config = load_configured_provider(workspace_root, label="run-experiment-batch")
    effective_ledger = effective_ledger_path(config, override=ledger_path)
    backend = select_execution_backend(
        backend_name,
        docker_image,
        docker_enable_gpu=docker_enable_gpu,
        docker_user=docker_user,
        docker_rootless_container_root=docker_rootless_container_root,
    )
    max_samples, max_prediction_samples, prediction_sample_policy = effective_execution_options(
        config,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
    )
    return run_experiment_batch_with_research_problem(
        batch,
        batches_root=Path(batches_root),
        runs_root=Path(runs_root),
        provider_config=provider_config,
        backend=backend,
        max_parallel_runs=max_parallel_runs,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
        ledger_path=effective_ledger,
    )


def run_evaluation_request_from_workspace(
    request_path: str | Path,
    *,
    workspace_root: str | Path,
    ledger_path: Path | None = None,
) -> dict[str, object]:
    """Run one request-gated Post-Run Evaluation through the configured backend."""

    from ml_autoresearch.evaluation_requests import _evaluation_id, validate_evaluation_request_file

    config = load_candidate_execution_config(workspace_root)
    effective_ledger = config.ledger_path if ledger_path is None else ledger_path
    if effective_ledger is None:
        raise CandidateExecutionConfigError("configure candidate_execution.ledger_path before executing next actions")
    backend = execution_backend_from_config(config)
    operation = backend.run_post_run_evaluation(request_path, runs_root=config.runs_root, ledger_path=effective_ledger)
    request = validate_evaluation_request_file(request_path)
    assert request.request_id is not None
    evaluation_id = _evaluation_id(request.request_id)
    metadata_path = config.runs_root / request.target_run_id / "outputs" / "evaluations" / evaluation_id / "evaluation_metadata.json"
    evaluation = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    return {
        "status": "completed",
        "executed": True,
        "action": "run_post_run_evaluation",
        "backend": operation.backend,
        "evaluation_id": evaluation_id,
        "evaluation": evaluation,
    }


def execute_ingested_next_action(root: str | Path, ingestion: dict[str, object]) -> dict[str, object]:
    """Execute at most one Harness-owned next action selected by Agent Handoff Ingestion."""

    root_path = Path(root)
    handoff_type = ingestion.get("handoff_type")
    next_action = ingestion.get("next_action")
    if handoff_type == "candidate_submission" and next_action == "run_candidate":
        return _execute_ingested_candidate_next_action(root_path, ingestion)
    if handoff_type == "experiment_batch_submission" and next_action == "run_experiment_batch":
        batch_path = required_relative_path(root_path, ingestion, "canonical_path")
        result = _run_ingested_experiment_batch(root_path, batch_path)
        return {
            "status": "completed",
            "executed": True,
            "action": "run_experiment_batch",
            "batch_id": result["batch_id"],
            "batch_dir": result["batch_dir"],
            "batch_status": result["status"],
            "runs": result["runs"],
        }
    if handoff_type == "evaluation_request" and next_action == "run_post_run_evaluation":
        request_path = required_relative_path(root_path, ingestion, "canonical_path")
        return run_evaluation_request_from_workspace(request_path, workspace_root=root_path)
    return {"status": "skipped", "executed": False, "reason": f"no executable Harness action for {handoff_type!r}"}


def required_relative_path(root: Path, payload: dict[str, object], field: str) -> Path:
    """Resolve a payload path that must stay inside the Research Workspace Root."""

    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ResearchLoopOperationError(f"ingestion result missing {field}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ResearchLoopOperationError(f"ingestion result {field} must be a relative path inside the Research Workspace Root")
    return root / path


def _run_ingested_experiment_batch(root: Path, batch_path: Path) -> dict[str, object]:
    config = load_candidate_execution_config(root)
    provider_config = resolve_configured_research_problem_provider(config)
    if provider_config is None:
        raise ResearchLoopOperationError("configure [research_problem] before executing Experiment Batch next actions")
    if config.ledger_path is None:
        raise ResearchLoopOperationError("configure candidate_execution.ledger_path before executing Experiment Batch next actions")
    from ml_autoresearch.batches import run_experiment_batch_with_research_problem

    return run_experiment_batch_with_research_problem(
        batch_path,
        batches_root=root / "batches",
        runs_root=config.runs_root,
        provider_config=provider_config,
        backend=execution_backend_from_config(config),
        max_parallel_runs=4,
        max_samples=config.max_samples,
        max_prediction_samples=config.max_prediction_samples,
        prediction_sample_policy=config.prediction_sample_policy,
        ledger_path=config.ledger_path,
    )


def _execute_ingested_candidate_next_action(root: Path, ingestion: dict[str, object]) -> dict[str, object]:
    from ml_autoresearch.runs import RunSubmission, submit_candidate, train_accepted_run_with_research_problem

    candidate_path = required_relative_path(root, ingestion, "canonical_path")
    config = load_candidate_execution_config(root)
    provider_config = resolve_configured_research_problem_provider(config)
    if provider_config is None:
        raise ResearchLoopOperationError("configure [research_problem] before executing candidate next actions")
    if config.ledger_path is None:
        raise ResearchLoopOperationError("configure candidate_execution.ledger_path before executing candidate next actions")
    backend = execution_backend_from_config(config)
    research_problem_registry = load_configured_research_problem_registry(root)
    previous_result = ingestion.get("next_action_result")
    run = None
    if isinstance(previous_result, dict) and previous_result.get("run_status") == "accepted":
        previous_run_dir = previous_result.get("run_dir")
        if isinstance(previous_run_dir, str) and previous_run_dir:
            previous_run_path = Path(previous_run_dir)
            previous_metadata_path = previous_run_path / "run_metadata.json"
            if previous_metadata_path.is_file():
                previous_metadata = json.loads(previous_metadata_path.read_text())
                previous_status = previous_metadata.get("status")
                if previous_status == RunStatus.ACCEPTED.value:
                    run = train_accepted_run_with_research_problem(
                        previous_run_path,
                        provider_config,
                        max_samples=config.max_samples,
                        max_prediction_samples=config.max_prediction_samples,
                        prediction_sample_policy=config.prediction_sample_policy,
                        backend=backend,
                        ledger_path=config.ledger_path,
                    )
                elif previous_status in {status.value for status in RunStatus}:
                    run = RunSubmission(
                        str(previous_metadata.get("run_id") or previous_run_path.name),
                        previous_run_path,
                        RunStatus(str(previous_status)),
                    )
                else:
                    raise ResearchLoopOperationError(
                        f"recorded accepted Run has unknown status {previous_status!r}: {previous_run_path}"
                    )
    if run is None:
        run = submit_candidate(
            candidate_path,
            config.runs_root,
            backend=backend,
            ledger_path=config.ledger_path,
            require_proposal=True,
            research_problem_registry=research_problem_registry,
        )
        if run.status.value == "accepted":
            run = train_accepted_run_with_research_problem(
                run.run_dir,
                provider_config,
                max_samples=config.max_samples,
                max_prediction_samples=config.max_prediction_samples,
                prediction_sample_policy=config.prediction_sample_policy,
                backend=backend,
                ledger_path=config.ledger_path,
            )
    return {
        "status": "completed",
        "executed": True,
        "action": "run_candidate",
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "run_status": run.status.value,
        "rejection_reason": run.rejection_reason,
        "failure_classification": run.failure_classification.value if run.failure_classification is not None else None,
    }
