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
)
from ml_autoresearch.capability_requests import CapabilityRequestError, create_capability_request
from ml_autoresearch.evaluation_requests import EvaluationRequestError, run_post_run_evaluation
from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, DockerBackend, ExecutionBackend, NativeBackend, validate_docker_gpu
from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, ResearchLedgerError, record_research_event
from ml_autoresearch.runs import RunStatus, get_best_runs, get_run_summary, list_runs

CLI_DEFAULT_MAX_ARTIFACT_SAMPLES = 12

app = typer.Typer(help="ML Autoresearch local Harness commands.")


@app.callback()
def root() -> None:
    """ML Autoresearch local Harness commands."""


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


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


@app.command("prepare-agent-boundary")
def prepare_agent_boundary_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-boundary.toml.")] = Path("."),
) -> None:
    """Prepare Agent Control Boundary snapshots, workspace directories, and pi-fort config."""

    try:
        result = prepare_agent_boundary(project_root)
    except (AgentBoundaryError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("autonomy-step")
def autonomy_step_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-boundary.toml.")] = Path("."),
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
        result = run_autonomy_step(project_root, agent_command=agent_command, execute_next_action=execute_next_action)
    except (AutonomyStepError, AgentBoundaryError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    typer.echo(format_autonomy_step_summary(result))
    if result.status in {"agent_failed", "ingestion_failed", "execution_failed"}:
        raise typer.Exit(1)


@app.command("run-autonomous-iteration")
def run_autonomous_iteration_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-boundary.toml and notification.toml.")] = Path("."),
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
        max_duration_seconds = parse_duration_seconds(max_duration) if max_duration is not None else None
        result = run_autonomous_iteration(
            project_root,
            agent_command=agent_command,
            max_steps=max_steps,
            max_duration_seconds=max_duration_seconds,
            notify_email=notify_email,
        )
    except (AutonomousIterationError, AutonomyStepError, AgentBoundaryError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    typer.echo(format_autonomous_iteration_summary(result))


@app.command("execute-next-action")
def execute_next_action_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-work/autonomy-step-result.json.")] = Path("."),
) -> None:
    """Execute the outstanding Harness-owned next action from the previous Autonomy Step."""

    try:
        result = execute_outstanding_next_action(project_root)
    except (AutonomyStepError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    typer.echo(format_autonomy_step_summary(result))
    if result.status == "execution_failed":
        raise typer.Exit(1)


@app.command("ingest-agent-handoff")
def ingest_agent_handoff_command(
    project_root: Annotated[
        Path, typer.Option(help="Project root containing agent-work handoff directories.")
    ] = Path("."),
) -> None:
    """Collect and ingest exactly one primary Agent Workspace handoff."""

    try:
        result = collect_agent_handoff(project_root)
    except (ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-candidate-submission")
def ingest_candidate_submission_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-work/submissions.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Candidate Submission into canonical candidates/."""

    try:
        result = ingest_candidate_submission(project_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-research-note")
def ingest_research_note_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-work/research-notes.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Research Note into canonical research-notes/."""

    try:
        result = ingest_research_note(project_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-capability-request")
def ingest_capability_request_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-work/capability-requests.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Capability Request into canonical capability-requests/."""

    try:
        result = ingest_capability_request(project_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-evaluation-request")
def ingest_evaluation_request_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-work/evaluation-requests.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Evaluation Request without executing it."""

    try:
        result = ingest_evaluation_request(project_root)
    except (AgentHandoffIngestionError, ResearchLedgerError, OSError) as exc:
        _exit_with_error(exc)
    _echo_json(result)


@app.command("ingest-campaign-report")
def ingest_campaign_report_command(
    project_root: Annotated[Path, typer.Option(help="Project root containing agent-work/campaign-reports.")] = Path("."),
) -> None:
    """Ingest one Agent Workspace Campaign Report into canonical campaign-reports/."""

    try:
        result = ingest_campaign_report(project_root)
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
    ledger_path: Annotated[
        Path,
        typer.Option(help="Append-only Research Ledger JSONL path."),
    ] = Path(CANONICAL_RESEARCH_LEDGER),
) -> None:
    """Validate an Evaluation Request and run a bounded Post-Run Evaluation."""

    try:
        result = run_post_run_evaluation(request, runs_root=runs_root, ledger_path=ledger_path)
    except (EvaluationRequestError, ResearchLedgerError, OSError) as exc:
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

    from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory

    try:
        manifest = validate_candidate_directory(candidate, require_proposal=require_proposal, require_readme=require_readme)
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


@app.command("run-candidate")
def run_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    runs_root: Annotated[Path, typer.Option(help="Directory where Harness Run directories are created.")],
    synthetic_fixture: Annotated[bool, typer.Option("--synthetic-fixture", help="Use deterministic generated contrail data.")] = False,
    data_root: Annotated[Path | None, typer.Option("--data-root", help="Local GVCCS Dataset root.")] = None,
    max_samples: Annotated[int | None, typer.Option("--max-samples", help="Bound the number of discovered GVCCS samples used.")] = None,
    max_prediction_samples: Annotated[
        int,
        typer.Option("--max-prediction-samples", help="Maximum number of qualitative prediction samples to write."),
    ] = 2,
    prediction_sample_policy: Annotated[
        Literal["first_n", "adjacent_and_scattered"],
        typer.Option("--prediction-sample-policy", help="Harness-owned qualitative Prediction Sample Policy."),
    ] = "first_n",
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
    ledger_path: Annotated[
        Path | None,
        typer.Option("--ledger-path", help="Append-only Research Ledger JSONL path. Defaults to runs_root sibling research-ledger.jsonl."),
    ] = None,
) -> None:
    """Validate, smoke-test, and synchronously run a Candidate Experiment."""

    if synthetic_fixture and data_root is not None:
        raise typer.BadParameter("choose either --synthetic-fixture or --data-root, not both")
    if daemonize:
        _daemonize_current_run_candidate(runs_root)
        return
    from ml_autoresearch.runs import run_candidate_with_gvccs_data, run_candidate_with_synthetic_fixture

    selected_backend = _select_backend(backend, docker_image, docker_enable_gpu, docker_user, docker_rootless_container_root)
    try:
        if synthetic_fixture:
            run = run_candidate_with_synthetic_fixture(
                candidate,
                runs_root,
                max_prediction_samples=max_prediction_samples,
                prediction_sample_policy=prediction_sample_policy,
                backend=selected_backend,
                ledger_path=ledger_path,
                require_proposal=require_proposal,
            )
        elif data_root is not None:
            run = run_candidate_with_gvccs_data(
                candidate,
                runs_root,
                data_root,
                max_samples=max_samples,
                max_prediction_samples=max_prediction_samples,
                prediction_sample_policy=prediction_sample_policy,
                backend=selected_backend,
                ledger_path=ledger_path,
                require_proposal=require_proposal,
            )
        else:
            raise typer.BadParameter("provide --data-root /path/to/gvccs or --synthetic-fixture")
    except (ResearchLedgerError, OSError) as exc:
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


@app.command("evaluate-run")
def evaluate_run_command(
    run: Annotated[Path, typer.Option("--run", help="Path to a completed source Run directory.")],
    split: Annotated[Literal["val"], typer.Option("--split", help="Run split to evaluate.")] = "val",
    backend: Annotated[Literal["native", "docker"], typer.Option("--backend", help="Post-Run Evaluation backend.")] = "docker",
    data_root: Annotated[Path | None, typer.Option("--data-root", help="Override GVCCS Dataset root from Run metadata.")] = None,
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
    except (EvaluationError, RuntimeError, ResearchLedgerError, OSError) as exc:
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
