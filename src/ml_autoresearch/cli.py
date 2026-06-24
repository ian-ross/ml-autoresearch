"""Command-line interface for ML Autoresearch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import typer

from ml_autoresearch.agent_boundary import AgentBoundaryError, prepare_agent_boundary
from ml_autoresearch.agent_handoffs import (
    AgentHandoffIngestionError,
    collect_agent_handoff,
    ingest_campaign_report,
    ingest_candidate_submission,
    ingest_capability_request,
    ingest_experiment_batch_submission,
    ingest_evaluation_request,
    ingest_research_note,
)
from ml_autoresearch.autonomous_iteration import (
    AutonomousIterationError,
    format_autonomous_iteration_summary,
    parse_duration_seconds,
    run_autonomous_iteration,
)
from ml_autoresearch.autonomy_step import (
    AutonomyStepError,
    execute_outstanding_next_action,
    format_autonomy_step_summary,
    run_autonomy_step,
)
from ml_autoresearch.campaign_controls import (
    CampaignControlError,
    record_campaign_pause,
    record_campaign_report_written,
    record_campaign_resume,
)
from ml_autoresearch.candidate_execution_config import CandidateExecutionConfigError
from ml_autoresearch.capability_requests import CapabilityRequestError, create_capability_request
from ml_autoresearch.evaluation_requests import EvaluationRequestError, run_post_run_evaluation
from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, DockerBackend, ExecutionBackend, NativeBackend, validate_docker_gpu
from ml_autoresearch.package_resources import PackageResourceError, stage_workspace_container_build_recipes
from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, ResearchLedgerError, record_research_event
from ml_autoresearch.runtime_images import (
    RuntimeImageError,
    build_runtime_images,
    require_runtime_image_validation,
    runtime_image_validation_skip_warning,
    validate_runtime_images,
)
from ml_autoresearch.runs import RunStatus, get_best_runs, get_run_summary, list_runs
from ml_autoresearch.batches import get_batch_summary, list_batches
from ml_autoresearch.setup import (
    SUPPORTED_BINARY_SEGMENTATION,
    WorkspaceSetupError,
    WorkspaceSetupRequest,
    infer_provider_module_from_pyproject,
    initialize_workspace,
)

CLI_DEFAULT_MAX_ARTIFACT_SAMPLES = 12

app = typer.Typer(help="ML Autoresearch local Harness commands.")


@app.callback()
def root() -> None:
    """ML Autoresearch local Harness commands."""


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _enforce_runtime_image_validation(command_name: str, workspace_root: Path, *, skip: bool) -> None:
    if skip:
        typer.echo(runtime_image_validation_skip_warning(command_name, workspace_root), err=True)
        return
    require_runtime_image_validation(workspace_root)


def _workspace_config_exists(workspace_root: Path) -> bool:
    return (Path(workspace_root).resolve() / "ml-autoresearch.toml").is_file()


def _workspace_root_for_evaluation_request(workspace_root: Path | None, runs_root: Path) -> Path:
    if workspace_root is not None:
        return workspace_root
    cwd = Path.cwd()
    if _workspace_config_exists(cwd):
        return cwd
    return runs_root.resolve().parent


def _select_backend(
    name: str,
    docker_image: str,
    docker_enable_gpu: bool = False,
    docker_user: str | None = None,
    docker_rootless_container_root: bool = False,
) -> ExecutionBackend:
    if name == "native":
        if docker_enable_gpu:
            raise typer.BadParameter("--docker-enable-gpu requires --backend docker")
        if docker_user is not None:
            raise typer.BadParameter("--docker-user requires --backend docker")
        if docker_rootless_container_root:
            raise typer.BadParameter("--docker-rootless-container-root requires --backend docker")
        return NativeBackend()
    if name == "docker":
        if docker_user is not None and docker_rootless_container_root:
            raise typer.BadParameter("choose either --docker-user or --docker-rootless-container-root, not both")
        return DockerBackend(
            docker_image,
            enable_gpu=docker_enable_gpu,
            container_user=docker_user,
            rootless_container_root=docker_rootless_container_root,
        )
    raise typer.BadParameter("backend must be native or docker")


def _load_configured_provider(
    workspace_root: Path,
    *,
    label: str = "run",
):
    from ml_autoresearch.candidate_execution_config import (
        CONFIG_FILENAME,
        CandidateExecutionConfigError,
        load_candidate_execution_config,
        resolve_configured_research_problem_provider,
    )

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

    resolved_data_root = _resolve_research_problem_data_root(provider_config.data_config, workspace_root=root)
    if resolved_data_root is not None:
        provider_config = resolve_configured_research_problem_provider(config, data_root_override=resolved_data_root)
        if provider_config is None:
            raise CandidateExecutionConfigError(f"configure [research_problem] in {config_path}")
    return config, provider_config


def _resolve_research_problem_data_root(configured_data_config: dict[str, object], *, workspace_root: Path) -> Path | None:
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


def _effective_execution_options(
    config,
    *,
    max_samples: int | None,
    max_prediction_samples: int | None,
    prediction_sample_policy: str | None,
) -> tuple[int | None, int, str]:
    return (
        config.max_samples if max_samples is None else max_samples,
        config.max_prediction_samples if max_prediction_samples is None else max_prediction_samples,
        config.prediction_sample_policy if prediction_sample_policy is None else prediction_sample_policy,
    )


def _effective_ledger_path(config, *, override: Path | None) -> Path:
    if override is not None:
        return override
    if config.ledger_path is None:
        raise CandidateExecutionConfigError(
            "configure candidate_execution.ledger_path in ml-autoresearch.toml or pass --ledger-path"
        )
    return config.ledger_path


def _echo_run(run) -> None:
    _echo_json(
        {
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "status": run.status.value,
            "rejection_reason": run.rejection_reason,
            "failure_classification": run.failure_classification.value if run.failure_classification is not None else None,
        }
    )


def _parse_event_fields(fields: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for field in fields:
        if "=" not in field:
            raise typer.BadParameter("event fields must be KEY=VALUE")
        key, value = field.split("=", 1)
        if not key:
            raise typer.BadParameter("event field keys must be non-empty")
        parsed[key] = value
    return parsed


def _exit_with_error(exc: BaseException) -> None:
    typer.echo(str(exc), err=True)
    raise typer.Exit(1) from exc


@app.command("setup")
def setup_command(
    workspace_root: Annotated[Path, typer.Option(help="Directory to initialize as the Research Workspace Root.")] = Path("."),
    problem_id: Annotated[str | None, typer.Option("--problem-id", help="Research Problem id, e.g. satellite_cloud_segmentation.")] = None,
    provider_module: Annotated[str | None, typer.Option("--provider-module", help="Python package module for starter trusted problem code.")] = None,
    problem_type: Annotated[str, typer.Option("--problem-type", help="Starter problem type to generate.")] = SUPPORTED_BINARY_SEGMENTATION,
    runs_root: Annotated[str | None, typer.Option("--runs-root", help="Directory where local Run artifacts will be written.")] = None,
    non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Use defaults/options without prompts for tests or automation.")] = False,
    reset_research_memory: Annotated[
        bool,
        typer.Option("--reset-research-memory", help="Explicitly truncate research-ledger.jsonl if it already exists."),
    ] = False,
) -> None:
    """Initialize a Research Problem Repository as a Research Workspace Root."""

    root = workspace_root.resolve()
    try:
        inferred_provider = infer_provider_module_from_pyproject(root)
    except WorkspaceSetupError as exc:
        _exit_with_error(exc)

    if non_interactive:
        effective_problem_id = problem_id or inferred_provider
        effective_provider = provider_module or inferred_provider
        effective_runs_root = runs_root or "runs"
    else:
        typer.echo("Research Workspace Setup will create missing workspace files conservatively.")
        effective_problem_id = typer.prompt("Research Problem id", default=problem_id or inferred_provider)
        inferred_message = f"Inferred provider module from pyproject.toml: {inferred_provider}"
        typer.echo(inferred_message)
        if provider_module is not None:
            effective_provider = provider_module
        elif typer.confirm("Use inferred provider module?", default=True):
            effective_provider = inferred_provider
        else:
            effective_provider = typer.prompt("Provider module", default=inferred_provider)
        effective_runs_root = typer.prompt(
            "Runs root (Run artifacts are normally not committed; use external storage for large/long-lived Runs)",
            default=runs_root or "runs",
        )

    typer.echo(
        "Runs root stores local Run artifacts. Keeping it inside the workspace is convenient; "
        "external storage can reduce repository/disk pressure. Run artifacts are normally not committed."
    )
    try:
        result = initialize_workspace(
            WorkspaceSetupRequest(
                workspace_root=root,
                problem_id=effective_problem_id,
                provider_module=effective_provider,
                problem_type=problem_type,
                runs_root=effective_runs_root,
                reset_research_memory=reset_research_memory,
            )
        )
    except (WorkspaceSetupError, OSError) as exc:
        _exit_with_error(exc)

    typer.echo(f"Initialized Research Workspace Root: {result.workspace_root}")
    typer.echo(f"Provider target: {result.provider_target}")
    typer.echo(f"Runs root: {result.runs_root}")
    typer.echo(f"Created {len(result.created)} path(s); skipped {len(result.skipped)} existing path(s).")
    if problem_type != SUPPORTED_BINARY_SEGMENTATION:
        typer.echo("Unsupported starter problem type generated TODO provider; implement it before real Runs.")
    else:
        typer.echo("Before real Runs, replace starter TODOs with trusted data adapters and problem-specific policy.")


@app.command("prepare-agent-boundary")
def prepare_agent_boundary_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing ml-autoresearch.toml.")] = Path("."),
    skip_runtime_image_validation: Annotated[
        bool,
        typer.Option("--skip-runtime-image-validation", help="Bypass the Runtime Image Validation Stamp check with a prominent warning."),
    ] = False,
) -> None:
    """Prepare Agent Control Boundary snapshots, workspace directories, and pi-fort config."""

    try:
        _enforce_runtime_image_validation("prepare-agent-boundary", workspace_root, skip=skip_runtime_image_validation)
        result = prepare_agent_boundary(workspace_root)
    except (RuntimeImageError, AgentBoundaryError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("stage-runtime-build-recipes")
def stage_runtime_build_recipes_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root for hidden runtime build state.")] = Path("."),
) -> None:
    """Stage packaged runtime build recipes into hidden workspace operational state."""

    try:
        result = stage_workspace_container_build_recipes(workspace_root.resolve())
    except (PackageResourceError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json({"destination": str(result.destination), "copied": [str(path) for path in result.copied]})


@app.command("build-runtime-images")
def build_runtime_images_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing ml-autoresearch.toml.")] = Path("."),
    update_config: Annotated[
        bool,
        typer.Option("--update-config", help="Explicitly update ml-autoresearch.toml with the built runtime image identities."),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute/--no-execute", help="Run Docker/Gondolin build commands instead of only staging recipes and metadata."),
    ] = True,
) -> None:
    """Build or prepare workspace-specific runtime images from packaged recipes."""

    try:
        result = build_runtime_images(workspace_root, update_config=update_config, execute=execute)
    except (RuntimeImageError, PackageResourceError, subprocess.CalledProcessError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result.model_dump())


@app.command("validate-runtime-images")
def validate_runtime_images_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing ml-autoresearch.toml.")] = Path("."),
) -> None:
    """Validate configured runtime image identities and write the validation stamp."""

    try:
        stamp = validate_runtime_images(workspace_root)
    except (RuntimeImageError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(stamp)


@app.command("autonomy-step")
def autonomy_step_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing ml-autoresearch.toml.")] = Path("."),
    agent_command: Annotated[
        str | None,
        typer.Option(
            "--agent-command",
            help="Agent command to invoke inside agent-work; defaults to [autonomy_step].agent_command or pi.",
        ),
    ] = None,
    execute_next_action: Annotated[
        bool,
        typer.Option(
            "--execute-next-action",
            help="After successful ingestion, execute at most one Harness-owned next action for executable handoffs.",
        ),
    ] = False,
) -> None:
    """Run one Autonomy Step: prepare boundary, invoke agent once, ingest one handoff, and optionally execute its next action."""

    try:
        result = run_autonomy_step(workspace_root, agent_command=agent_command, execute_next_action=execute_next_action)
    except (AutonomyStepError, AgentBoundaryError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    typer.echo(format_autonomy_step_summary(result))
    if result.status in {"agent_failed", "ingestion_failed", "execution_failed"}:
        raise typer.Exit(1)


@app.command("run-autonomous-iteration")
def run_autonomous_iteration_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing ml-autoresearch.toml.")] = Path("."),
    skip_runtime_image_validation: Annotated[
        bool,
        typer.Option("--skip-runtime-image-validation", help="Bypass the Runtime Image Validation Stamp check with a prominent warning."),
    ] = False,
    agent_command: Annotated[
        str | None,
        typer.Option(
            "--agent-command",
            help="Agent command to invoke inside agent-work; defaults to [autonomy_step].agent_command or pi.",
        ),
    ] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps", help="Maximum Autonomy Steps to complete before stopping.")] = None,
    max_duration: Annotated[
        str | None,
        typer.Option("--max-duration", help="Maximum elapsed duration before starting another step: N, Ns, Nm, or Nh."),
    ] = None,
    notify_email: Annotated[str, typer.Option("--notify-email", help="Email address to notify when the loop completes.")] = ...,
) -> None:
    """Run a bounded autonomous iteration loop and send a completion email."""

    try:
        _enforce_runtime_image_validation("run-autonomous-iteration", workspace_root, skip=skip_runtime_image_validation)
        max_duration_seconds = parse_duration_seconds(max_duration) if max_duration is not None else None
        result = run_autonomous_iteration(
            workspace_root,
            agent_command=agent_command,
            max_steps=max_steps,
            max_duration_seconds=max_duration_seconds,
            notify_email=notify_email,
        )
    except (RuntimeImageError, AutonomousIterationError, AutonomyStepError, AgentBoundaryError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    typer.echo(format_autonomous_iteration_summary(result))


@app.command("execute-next-action")
def execute_next_action_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing agent-work/autonomy-step-result.json.")] = Path("."),
) -> None:
    """Execute the outstanding Harness-owned next action from the previous Autonomy Step."""

    try:
        result = execute_outstanding_next_action(workspace_root)
    except (AutonomyStepError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    typer.echo(format_autonomy_step_summary(result))
    if result.status == "execution_failed":
        raise typer.Exit(1)


@app.command("ingest-agent-handoff")
def ingest_agent_handoff_command(
    workspace_root: Annotated[
        Path, typer.Option(help="Research Workspace Root containing agent-work handoff directories.")
    ] = Path("."),
) -> None:
    """Collect and ingest exactly one primary Agent Workspace handoff."""

    try:
        result = collect_agent_handoff(workspace_root)
    except (ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-candidate-submission")
def ingest_candidate_submission_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing agent-work/submissions.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Candidate Submission into canonical candidates/."""

    try:
        result = ingest_candidate_submission(workspace_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-experiment-batch-submission")
def ingest_experiment_batch_submission_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing agent-work/batch-submissions.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Experiment Batch Submission into canonical experiment-batches/."""

    try:
        result = ingest_experiment_batch_submission(workspace_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)
    if result.get("status") == "ingestion_failed":
        raise typer.Exit(1)


@app.command("ingest-research-note")
def ingest_research_note_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing agent-work/research-notes.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Research Note into canonical research-notes/."""

    try:
        result = ingest_research_note(workspace_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-capability-request")
def ingest_capability_request_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing agent-work/capability-requests.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Capability Request into canonical capability-requests/."""

    try:
        result = ingest_capability_request(workspace_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-evaluation-request")
def ingest_evaluation_request_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing agent-work/evaluation-requests.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Evaluation Request without executing it."""

    try:
        result = ingest_evaluation_request(workspace_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-campaign-report")
def ingest_campaign_report_command(
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing agent-work/campaign-reports.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Campaign Report into canonical campaign-reports/."""

    try:
        result = ingest_campaign_report(workspace_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("create-capability-request")
def create_capability_request_command(
    request: Annotated[Path, typer.Option(help="Path to a YAML Capability Request file.")],
    ledger_path: Annotated[
        Path,
        typer.Option(help="Append-only Research Ledger JSONL path."),
    ] = Path(CANONICAL_RESEARCH_LEDGER),
) -> None:
    """Validate a Capability Request and record a creation event for human review."""

    try:
        result = create_capability_request(request, ledger_path=ledger_path)
    except (CapabilityRequestError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("run-post-run-evaluation")
def run_post_run_evaluation_command(
    request: Annotated[Path, typer.Option(help="Path to a YAML Evaluation Request file.")],
    runs_root: Annotated[Path, typer.Option(help="Directory containing local Harness Run directories.")],
    workspace_root: Annotated[
        Path | None,
        typer.Option("--workspace-root", help="Research Workspace Root containing ml-autoresearch.toml; defaults to runs_root parent when present."),
    ] = None,
    skip_runtime_image_validation: Annotated[
        bool,
        typer.Option("--skip-runtime-image-validation", help="Bypass the Runtime Image Validation Stamp check with a prominent warning."),
    ] = False,
    ledger_path: Annotated[
        Path,
        typer.Option(help="Append-only Research Ledger JSONL path."),
    ] = Path(CANONICAL_RESEARCH_LEDGER),
) -> None:
    """Validate an Evaluation Request and run a bounded Post-Run Evaluation."""

    try:
        validation_root = _workspace_root_for_evaluation_request(workspace_root, runs_root)
        if _workspace_config_exists(validation_root):
            from ml_autoresearch.candidate_execution_config import execution_backend_from_config, load_candidate_execution_config
            from ml_autoresearch.evaluation_requests import _evaluation_id, validate_evaluation_request_file

            config = load_candidate_execution_config(validation_root)
            if config.backend == "docker":
                _enforce_runtime_image_validation("run-post-run-evaluation", validation_root, skip=skip_runtime_image_validation)
                effective_ledger_path = config.ledger_path or ledger_path
                operation = execution_backend_from_config(config).run_post_run_evaluation(
                    request,
                    runs_root=config.runs_root,
                    ledger_path=effective_ledger_path,
                )
                evaluation_request = validate_evaluation_request_file(request)
                assert evaluation_request.request_id is not None
                evaluation_id = _evaluation_id(evaluation_request.request_id)
                metadata_path = (
                    config.runs_root
                    / evaluation_request.target_run_id
                    / "outputs"
                    / "evaluations"
                    / evaluation_id
                    / "evaluation_metadata.json"
                )
                evaluation = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
                _echo_json(
                    {
                        "status": "completed",
                        "backend": operation.backend,
                        "evaluation_id": evaluation_id,
                        "evaluation": evaluation,
                    }
                )
                return
        result = run_post_run_evaluation(request, runs_root=runs_root, ledger_path=ledger_path)
    except (RuntimeImageError, CandidateExecutionConfigError, EvaluationRequestError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("record-campaign-report")
def record_campaign_report_command(
    report_path: Annotated[Path, typer.Option(help="Path to the Campaign Report artifact.")],
    ledger_path: Annotated[
        Path,
        typer.Option(help="Append-only Research Ledger JSONL path."),
    ] = Path(CANONICAL_RESEARCH_LEDGER),
) -> None:
    """Record a validated campaign_report_written Research Ledger event."""

    try:
        result = record_campaign_report_written(report_path, ledger_path=ledger_path)
    except (CampaignControlError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("pause-campaign")
def pause_campaign_command(
    reason: Annotated[str, typer.Option(help="Approved Campaign Pause Condition value.")],
    report_path: Annotated[Path | None, typer.Option(help="Optional Campaign Report path for this pause.")] = None,
    ledger_path: Annotated[
        Path,
        typer.Option(help="Append-only Research Ledger JSONL path."),
    ] = Path(CANONICAL_RESEARCH_LEDGER),
) -> None:
    """Record a validated campaign_paused Research Ledger event."""

    try:
        result = record_campaign_pause(reason, report_path=report_path, ledger_path=ledger_path)
    except (CampaignControlError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("resume-campaign")
def resume_campaign_command(
    reason: Annotated[str, typer.Option(help="Human-review reason for resuming the campaign.")] = "human_review_complete",
    report_path: Annotated[Path | None, typer.Option(help="Optional Campaign Report path documenting the resume decision.")] = None,
    ledger_path: Annotated[
        Path,
        typer.Option(help="Append-only Research Ledger JSONL path."),
    ] = Path(CANONICAL_RESEARCH_LEDGER),
) -> None:
    """Record a campaign_resumed Research Ledger event after human review."""

    try:
        result = record_campaign_resume(reason, report_path=report_path, ledger_path=ledger_path)
    except (CampaignControlError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("record-research-event")
def record_research_event_command(
    event_type: Annotated[str, typer.Option(help="Research Ledger event type to record.")],
    ledger_path: Annotated[
        Path,
        typer.Option(help="Append-only Research Ledger JSONL path."),
    ] = Path(CANONICAL_RESEARCH_LEDGER),
    field: Annotated[
        list[str] | None,
        typer.Option("--field", help="Event field as KEY=VALUE. Repeat for multiple fields."),
    ] = None,
) -> None:
    """Validate and append one structured event to research-ledger.jsonl."""

    try:
        event = record_research_event(event_type, _parse_event_fields(field or []), ledger_path=ledger_path)
    except (ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(event)


def _daemonize_current_run_candidate(runs_root: Path) -> None:
    """Re-exec the current run-candidate command in a detached child process."""

    daemon_logs = runs_root / "daemon_logs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = daemon_logs / f"run_candidate_{timestamp}.log"
    _daemonize_current_command(log_path)


def _daemonize_current_evaluate_run(run_dir: Path) -> None:
    """Re-exec the current evaluate-run command in a detached child process."""

    daemon_logs = run_dir / "outputs" / "evaluation_daemon_logs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = daemon_logs / f"evaluate_run_{timestamp}.log"
    _daemonize_current_command(log_path)


def _daemonize_current_command(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_args = [arg for arg in sys.argv[1:] if arg != "--daemonize"]
    command = [sys.executable, "-m", "ml_autoresearch.cli", *child_args]
    with log_path.open("ab") as log_file, Path(os.devnull).open("rb") as stdin:
        process = subprocess.Popen(
            command,
            stdin=stdin,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    _echo_json({"status": "daemonized", "pid": process.pid, "log_path": str(log_path), "command": command})


@app.command("validate-candidate")
def validate_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing optional ml-autoresearch.toml Research Problem provider config.")] = Path("."),
    require_proposal: Annotated[
        bool,
        typer.Option(
            "--require-proposal/--no-require-proposal",
            help="Require a candidate-local PROPOSAL.md during static validation.",
        ),
    ] = True,
    require_readme: Annotated[
        bool,
        typer.Option(
            "--require-readme/--no-require-readme",
            help="Require a candidate-local README.md during static validation.",
        ),
    ] = False,
) -> None:
    """Statically validate a Candidate Experiment contract without importing or executing model code."""

    from ml_autoresearch.candidate_execution_config import CandidateExecutionConfigError, load_configured_research_problem_registry
    from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
    from ml_autoresearch.research_problems import (
        ResearchProblemProviderLoadError,
        legacy_smoke_research_problem_registry,
    )

    try:
        try:
            registry = load_configured_research_problem_registry(workspace_root)
            if registry is None:
                registry = legacy_smoke_research_problem_registry()
        except (CandidateExecutionConfigError, ResearchProblemProviderLoadError):
            registry = legacy_smoke_research_problem_registry()
        manifest = validate_candidate_directory(
            candidate,
            require_proposal=require_proposal,
            require_readme=require_readme,
            research_problem_registry=registry,
        )
    except (CandidateValidationError, OSError) as exc:
        _echo_json({"status": "invalid", "reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json({"status": "valid", "manifest": manifest.model_dump(mode="json")})


@app.command("submit-candidate")
def submit_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    runs_root: Annotated[Path, typer.Option(help="Directory where Harness Run directories are created.")],
    backend: Annotated[Literal["native", "docker"], typer.Option("--backend", help="Candidate Execution Boundary backend.")] = "native",
    docker_image: Annotated[str, typer.Option("--docker-image", help="Docker runner image for --backend docker.")] = DEFAULT_DOCKER_IMAGE,
    docker_enable_gpu: Annotated[
        bool,
        typer.Option("--docker-enable-gpu", help="Opt in to Docker GPU access by passing --gpus all to Docker runs."),
    ] = False,
    docker_user: Annotated[
        str | None,
        typer.Option(
            "--docker-user",
            help="Container uid:gid for Docker runs. May create host artifacts owned by a remapped uid on rootless/userns Docker.",
        ),
    ] = None,
    docker_rootless_container_root: Annotated[
        bool,
        typer.Option(
            "--docker-rootless-container-root",
            help="Force rootless Docker ownership mode: run as container root, which maps to the invoking host user and preserves output ownership.",
        ),
    ] = False,
    require_proposal: Annotated[
        bool,
        typer.Option(
            "--require-proposal/--no-require-proposal",
            help="Require a local PROPOSAL.md during candidate validation.",
        ),
    ] = True,
    ledger_path: Annotated[
        Path | None,
        typer.Option("--ledger-path", help="Append-only Research Ledger JSONL path. Defaults to runs_root sibling research-ledger.jsonl."),
    ] = None,
) -> None:
    """Validate a local Candidate Experiment and create a Run."""

    from ml_autoresearch.runs import submit_candidate

    try:
        run = submit_candidate(
            candidate,
            runs_root,
            backend=_select_backend(backend, docker_image, docker_enable_gpu, docker_user, docker_rootless_container_root),
            ledger_path=ledger_path,
            require_proposal=require_proposal,
        )
    except (ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_run(run)
    if run.status in {RunStatus.REJECTED, RunStatus.SMOKE_FAILED}:
        raise typer.Exit(1)


@app.command("run-experiment-batch")
def run_experiment_batch_command(
    batch: Annotated[Path, typer.Option(help="Path to a local Experiment Batch directory.")],
    batches_root: Annotated[Path, typer.Option(help="Directory where Experiment Batch artifact directories are created.")],
    runs_root: Annotated[Path, typer.Option(help="Directory where Harness Run directories are created.")],
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing ml-autoresearch.toml Research Problem provider config.")] = Path("."),
    max_samples: Annotated[int | None, typer.Option("--max-samples", help="Bound the number of Research Problem samples used.")] = None,
    max_parallel_runs: Annotated[int, typer.Option("--max-parallel-runs", help="Harness-owned parallel Run cap, capped at 4.")] = 4,
    max_prediction_samples: Annotated[
        int | None,
        typer.Option("--max-prediction-samples", help="Maximum number of qualitative prediction samples to write per Run."),
    ] = None,
    prediction_sample_policy: Annotated[
        Literal["first_n", "adjacent_and_scattered"] | None,
        typer.Option("--prediction-sample-policy", help="Harness-owned qualitative Prediction Sample Policy."),
    ] = None,
    backend: Annotated[Literal["native", "docker"], typer.Option("--backend", help="Candidate Execution Boundary backend.")] = "docker",
    docker_image: Annotated[str, typer.Option("--docker-image", help="Docker runner image for --backend docker.")] = DEFAULT_DOCKER_IMAGE,
    docker_enable_gpu: Annotated[
        bool,
        typer.Option("--docker-enable-gpu", help="Opt in to Docker GPU access by passing --gpus all to Docker runs."),
    ] = False,
    docker_user: Annotated[str | None, typer.Option("--docker-user", help="Container uid:gid for Docker runs.")] = None,
    docker_rootless_container_root: Annotated[
        bool,
        typer.Option("--docker-rootless-container-root", help="Run as container root on rootless Docker to preserve output ownership."),
    ] = False,
    ledger_path: Annotated[
        Path | None,
        typer.Option("--ledger-path", help="Append-only Research Ledger JSONL path. Overrides ml-autoresearch.toml."),
    ] = None,
) -> None:
    """Validate and synchronously run a bounded Experiment Batch."""

    from ml_autoresearch.batches import (
        ExperimentBatchError,
        run_experiment_batch_with_research_problem,
    )
    from ml_autoresearch.research_problems import ResearchProblemProviderLoadError

    selected_backend = _select_backend(backend, docker_image, docker_enable_gpu, docker_user, docker_rootless_container_root)
    try:
        config, provider_config = _load_configured_provider(workspace_root, label="run-experiment-batch")
        effective_ledger_path = _effective_ledger_path(config, override=ledger_path)
        max_samples, max_prediction_samples, prediction_sample_policy = _effective_execution_options(
            config,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        )
        result = run_experiment_batch_with_research_problem(
            batch,
            batches_root=batches_root,
            runs_root=runs_root,
            provider_config=provider_config,
            backend=selected_backend,
            max_parallel_runs=max_parallel_runs,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
            ledger_path=effective_ledger_path,
        )
    except (
        CandidateExecutionConfigError,
        ExperimentBatchError,
        ResearchLedgerError,
        ResearchProblemProviderLoadError,
        OSError,
    ) as exc:
        _exit_with_error(exc)
    _echo_json(result)
    if result.get("status") != "completed":
        raise typer.Exit(1)


@app.command("run-candidate")
def run_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    runs_root: Annotated[
        Path | None,
        typer.Option(help="Directory where Harness Run directories are created. Defaults to ml-autoresearch.toml runs_root."),
    ] = None,
    workspace_root: Annotated[Path, typer.Option(help="Research Workspace Root containing ml-autoresearch.toml Research Problem provider config.")] = Path("."),
    max_samples: Annotated[int | None, typer.Option("--max-samples", help="Bound the number of Research Problem samples used.")] = None,
    max_prediction_samples: Annotated[
        int | None,
        typer.Option("--max-prediction-samples", help="Maximum number of qualitative prediction samples to write."),
    ] = None,
    prediction_sample_policy: Annotated[
        Literal["first_n", "adjacent_and_scattered"] | None,
        typer.Option("--prediction-sample-policy", help="Harness-owned qualitative Prediction Sample Policy."),
    ] = None,
    backend: Annotated[Literal["native", "docker"], typer.Option("--backend", help="Candidate Execution Boundary backend.")] = "docker",
    docker_image: Annotated[str, typer.Option("--docker-image", help="Docker runner image for --backend docker.")] = DEFAULT_DOCKER_IMAGE,
    docker_enable_gpu: Annotated[
        bool,
        typer.Option("--docker-enable-gpu", help="Opt in to Docker GPU access by passing --gpus all to Docker runs."),
    ] = False,
    docker_user: Annotated[
        str | None,
        typer.Option(
            "--docker-user",
            help="Container uid:gid for Docker runs. May create host artifacts owned by a remapped uid on rootless/userns Docker.",
        ),
    ] = None,
    docker_rootless_container_root: Annotated[
        bool,
        typer.Option(
            "--docker-rootless-container-root",
            help="Force rootless Docker ownership mode: run as container root, which maps to the invoking host user and preserves output ownership.",
        ),
    ] = False,
    require_proposal: Annotated[
        bool,
        typer.Option(
            "--require-proposal/--no-require-proposal",
            help="Require a candidate-local PROPOSAL.md before execution. Use --no-require-proposal for manual compatibility.",
        ),
    ] = True,
    daemonize: Annotated[
        bool,
        typer.Option("--daemonize", help="Start the Candidate Experiment Run in a detached background process and return immediately."),
    ] = False,
    skip_runtime_image_validation: Annotated[
        bool,
        typer.Option("--skip-runtime-image-validation", help="Bypass the Runtime Image Validation Stamp check with a prominent warning."),
    ] = False,
    ledger_path: Annotated[
        Path | None,
        typer.Option("--ledger-path", help="Append-only Research Ledger JSONL path. Overrides ml-autoresearch.toml."),
    ] = None,
) -> None:
    """Validate, smoke-test, and synchronously run a Candidate Experiment."""

    from ml_autoresearch.research_problems import ResearchProblemProviderLoadError
    from ml_autoresearch.runs import run_candidate_with_research_problem

    try:
        if backend == "docker":
            _enforce_runtime_image_validation("run-candidate", workspace_root, skip=skip_runtime_image_validation)
        config, provider_config = _load_configured_provider(workspace_root, label="run-candidate")
        effective_runs_root = config.runs_root if runs_root is None else runs_root
        effective_ledger_path = _effective_ledger_path(config, override=ledger_path)
        if daemonize:
            _daemonize_current_run_candidate(effective_runs_root)
            return
        selected_backend = _select_backend(backend, docker_image, docker_enable_gpu, docker_user, docker_rootless_container_root)
        max_samples, max_prediction_samples, prediction_sample_policy = _effective_execution_options(
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
            backend=selected_backend,
            ledger_path=effective_ledger_path,
            require_proposal=require_proposal,
        )
    except (RuntimeImageError, CandidateExecutionConfigError, ResearchLedgerError, ResearchProblemProviderLoadError, OSError) as exc:
        _exit_with_error(exc)
    _echo_run(run)
    if run.status != RunStatus.COMPLETED:
        raise typer.Exit(1)


def _echo_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        typer.echo("No local Runs found.")
        return
    typer.echo("run_id\tstatus\tval/dice\treason")
    for row in rows:
        metrics = row.get("metrics")
        dice = metrics.get("val/dice") if isinstance(metrics, dict) else ""
        typer.echo(
            f"{row.get('run_id', '')}\t{row.get('status', '')}\t{dice}\t{row.get('reason') or row.get('error', '')}"
        )


def _echo_batch_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        typer.echo("No local Experiment Batches found.")
        return
    typer.echo("batch_id\tstatus\truns")
    for row in rows:
        runs = row.get("runs")
        run_count = len(runs) if isinstance(runs, list) else ""
        typer.echo(f"{row.get('batch_id', '')}\t{row.get('status', '')}\t{run_count}")


@app.command("evaluate-run")
def evaluate_run_command(
    run: Annotated[Path, typer.Option("--run", help="Path to a completed source Run directory.")],
    split: Annotated[Literal["val"], typer.Option("--split", help="Run split to evaluate.")] = "val",
    backend: Annotated[Literal["native", "docker"], typer.Option("--backend", help="Post-Run Evaluation backend.")] = "docker",
    data_root: Annotated[Path | None, typer.Option("--data-root", help="Override Research Problem data root from Run metadata.")] = None,
    max_artifact_samples: Annotated[
        int,
        typer.Option("--max-artifact-samples", help="Maximum selected diagnostic samples to write as visual artifacts."),
    ] = CLI_DEFAULT_MAX_ARTIFACT_SAMPLES,
    docker_image: Annotated[str, typer.Option("--docker-image", help="Docker runner image for --backend docker.")] = DEFAULT_DOCKER_IMAGE,
    docker_enable_gpu: Annotated[
        bool,
        typer.Option("--docker-enable-gpu", help="Opt in to Docker GPU access by passing --gpus all to Docker evaluations."),
    ] = False,
    docker_user: Annotated[
        str | None,
        typer.Option("--docker-user", help="Container uid:gid for Docker evaluations."),
    ] = None,
    docker_rootless_container_root: Annotated[
        bool,
        typer.Option("--docker-rootless-container-root", help="Run as container root on rootless Docker to preserve output ownership."),
    ] = False,
    daemonize: Annotated[
        bool,
        typer.Option("--daemonize", help="Start the Post-Run Evaluation in a detached background process and return immediately."),
    ] = False,
    workspace_root: Annotated[
        Path | None,
        typer.Option("--workspace-root", help="Research Workspace Root containing ml-autoresearch.toml; defaults to the Run's runs_root parent when present."),
    ] = None,
    skip_runtime_image_validation: Annotated[
        bool,
        typer.Option("--skip-runtime-image-validation", help="Bypass the Runtime Image Validation Stamp check with a prominent warning."),
    ] = False,
    ledger_path: Annotated[
        Path | None,
        typer.Option("--ledger-path", help="Append-only Research Ledger JSONL path. Defaults to runs_root sibling research-ledger.jsonl."),
    ] = None,
) -> None:
    """Evaluate a completed Run without retraining and write run-scoped artifacts."""

    if daemonize:
        _daemonize_current_evaluate_run(run)
        return
    from ml_autoresearch.evaluations import EvaluationError, default_evaluation_ledger_path, evaluate_run

    selected_backend = _select_backend(backend, docker_image, docker_enable_gpu, docker_user, docker_rootless_container_root)
    try:
        if backend == "docker":
            validation_root = workspace_root if workspace_root is not None else run.resolve().parent.parent
            if _workspace_config_exists(validation_root):
                _enforce_runtime_image_validation("evaluate-run", validation_root, skip=skip_runtime_image_validation)
        resolved_ledger_path = ledger_path if ledger_path is not None else default_evaluation_ledger_path(run)
        if backend == "native":
            result = evaluate_run(
                run,
                split=split,
                backend="native",
                data_root=data_root,
                max_artifact_samples=max_artifact_samples,
                ledger_path=resolved_ledger_path,
            )
            _echo_json({"evaluation_id": result.evaluation_id, "evaluation_dir": str(result.evaluation_dir), "status": result.status})
            return
        selected_backend.evaluate_run(run, data_root=data_root, max_artifact_samples=max_artifact_samples)
        ledger_events = _record_latest_docker_evaluation(run, resolved_ledger_path)
    except (RuntimeImageError, EvaluationError, RuntimeError, ResearchLedgerError, OSError) as exc:
        _echo_json({"status": "failed", "failure_reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json({"status": "completed", "backend": "docker", "run_dir": str(run), "ledger_events": ledger_events})


def _record_latest_docker_evaluation(run_dir: Path, ledger_path: Path) -> list[dict[str, object]]:
    from ml_autoresearch.evaluations import (
        EvaluationError,
        record_manual_evaluation_completed,
        record_manual_evaluation_requested,
    )

    metadata_paths = sorted((run_dir / "outputs" / "evaluations").glob("eval_*/evaluation_metadata.json"))
    if not metadata_paths:
        raise EvaluationError("Docker evaluation completed without evaluation metadata")
    metadata_path = metadata_paths[-1]
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "completed":
        raise EvaluationError(f"Docker evaluation did not complete successfully: {metadata.get('status')}")
    evaluation_id = metadata.get("evaluation_id")
    source_run = metadata.get("source_run")
    run_id = source_run.get("run_id") if isinstance(source_run, dict) else None
    if not isinstance(evaluation_id, str) or not isinstance(run_id, str):
        raise EvaluationError("Docker evaluation metadata is missing evaluation_id or source_run.run_id")
    request_path = metadata_path.parent / "evaluation_request.json"
    if not request_path.is_file():
        raise EvaluationError("Docker evaluation completed without evaluation_request.json")
    requested = record_manual_evaluation_requested(
        ledger_path=ledger_path,
        evaluation_id=evaluation_id,
        request_path=request_path,
        run_id=run_id,
    )
    completed = record_manual_evaluation_completed(
        ledger_path=ledger_path,
        evaluation_id=evaluation_id,
        evaluation_request_id=f"manual_{evaluation_id}",
        run_id=run_id,
        artifact_metadata_path=metadata_path,
    )
    return [requested, completed]


@app.command("validate-docker-gpu")
def validate_docker_gpu_command(
    docker_image: Annotated[str, typer.Option("--docker-image", help="Docker runner image to validate.")] = DEFAULT_DOCKER_IMAGE,
) -> None:
    """Validate PyTorch/CUDA/GPU visibility inside the Docker runner image."""

    completed = validate_docker_gpu(docker_image)
    if completed.stdout:
        typer.echo(completed.stdout.rstrip())
    if completed.stderr:
        typer.echo(completed.stderr.rstrip(), err=True)
    if completed.returncode != 0:
        raise typer.Exit(completed.returncode)


@app.command("list-batches")
def list_batches_command(
    batches_root: Annotated[Path, typer.Option(help="Directory containing local Experiment Batch artifact directories.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List local Experiment Batches from the local batches/ artifact tree."""

    rows = list_batches(batches_root)
    if json_output:
        _echo_json(rows)
    else:
        _echo_batch_table(rows)


@app.command("batch-summary")
def batch_summary_command(
    batches_root: Annotated[Path, typer.Option(help="Directory containing local Experiment Batch artifact directories.")],
    batch_id: Annotated[str, typer.Option(help="Experiment Batch identifier to inspect.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Inspect one local Experiment Batch summary."""

    summary = get_batch_summary(batches_root, batch_id)
    if json_output:
        _echo_json(summary)
    else:
        _echo_batch_table([summary])
    if summary.get("status") in {"missing", "corrupt", "missing_metadata"}:
        raise typer.Exit(1)


@app.command("list-runs")
def list_runs_command(
    runs_root: Annotated[Path, typer.Option(help="Directory containing local Harness Run directories.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List local Runs from the local runs/ artifact tree."""

    rows = list_runs(runs_root)
    if json_output:
        _echo_json(rows)
    else:
        _echo_table(rows)


def _run_summary_command(runs_root: Path, run_id: str, json_output: bool) -> None:
    summary = get_run_summary(runs_root, run_id)
    if json_output:
        _echo_json(summary)
    else:
        _echo_table([summary])
    if summary.get("status") in {"missing", "corrupt", "missing_metadata"}:
        raise typer.Exit(1)


@app.command("run-summary")
def run_summary_command(
    runs_root: Annotated[Path, typer.Option(help="Directory containing local Harness Run directories.")],
    run_id: Annotated[str, typer.Option(help="Run identifier to inspect.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Inspect one local Run summary without MLflow."""

    _run_summary_command(runs_root, run_id, json_output)


@app.command("get-run-summary")
def get_run_summary_command(
    runs_root: Annotated[Path, typer.Option(help="Directory containing local Harness Run directories.")],
    run_id: Annotated[str, typer.Option(help="Run identifier to inspect.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Alias for run-summary."""

    _run_summary_command(runs_root, run_id, json_output)


@app.command("get-best-runs")
def get_best_runs_command(
    runs_root: Annotated[Path, typer.Option(help="Directory containing local Harness Run directories.")],
    metric: Annotated[str, typer.Option(help="Metric key used for ranking local Runs.")] = "val/dice",
    limit: Annotated[int | None, typer.Option(help="Maximum number of ranked Runs to print.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Identify best completed local Runs by val/dice by default."""

    rows = get_best_runs(runs_root, metric=metric, limit=limit)
    if json_output:
        _echo_json(rows)
    else:
        _echo_table(rows)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
