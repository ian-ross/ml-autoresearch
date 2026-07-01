"""Agent-safe command-line interface for ML Autoresearch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from ml_autoresearch.batches import get_batch_summary, list_batches
from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.capability_requests import (
    CapabilityRequestError,
    validate_capability_request_file,
    write_capability_request_file,
)
from ml_autoresearch.runs import get_best_runs, get_run_summary, list_runs

DEFAULT_AGENT_RUNS_ROOT = Path("/history/runs")
DEFAULT_AGENT_BATCHES_ROOT = Path("/history/batches")
_AGENT_RUNS_ROOT_ENV = "ML_AUTORESEARCH_AGENT_RUNS_ROOT"
_AGENT_BATCHES_ROOT_ENV = "ML_AUTORESEARCH_AGENT_BATCHES_ROOT"

app = typer.Typer(
    help=(
        "Agent-safe ML Autoresearch observation and static-preparation commands. "
        "This wrapper cannot run Candidate Experiments."
    )
)


@app.callback()
def root() -> None:
    """Agent-safe commands only; cannot run Candidate Experiments."""


def _resolve_runs_root(runs_root: Path | None) -> Path:
    if runs_root is not None:
        return runs_root
    return Path(os.environ.get(_AGENT_RUNS_ROOT_ENV, str(DEFAULT_AGENT_RUNS_ROOT)))


def _resolve_batches_root(batches_root: Path | None) -> Path:
    if batches_root is not None:
        return batches_root
    return Path(os.environ.get(_AGENT_BATCHES_ROOT_ENV, str(DEFAULT_AGENT_BATCHES_ROOT)))


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


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


@app.command("list-batches")
def list_batches_command(
    batches_root: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing local Experiment Batch artifact directories. Defaults to /history/batches inside the Agent Control Boundary."
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List prior local Experiment Batches from the read-only Research History."""

    rows = list_batches(_resolve_batches_root(batches_root))
    if json_output:
        _echo_json(rows)
    else:
        _echo_batch_table(rows)


@app.command("batch-summary")
def batch_summary_command(
    batch_id: Annotated[str, typer.Option(help="Experiment Batch identifier to inspect.")],
    batches_root: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing local Experiment Batch artifact directories. Defaults to /history/batches inside the Agent Control Boundary."
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Inspect one prior local Experiment Batch summary from the read-only Research History."""

    summary = get_batch_summary(_resolve_batches_root(batches_root), batch_id)
    if json_output:
        _echo_json(summary)
    else:
        _echo_batch_table([summary])
    if summary.get("status") in {"missing", "corrupt", "missing_metadata"}:
        raise typer.Exit(1)


@app.command("list-runs")
def list_runs_command(
    runs_root: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing local Harness Run directories. Defaults to /history/runs inside the Agent Control Boundary."
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List prior Runs from the read-only Research History."""

    rows = list_runs(_resolve_runs_root(runs_root))
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
    run_id: Annotated[str, typer.Option(help="Run identifier to inspect.")],
    runs_root: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing local Harness Run directories. Defaults to /history/runs inside the Agent Control Boundary."
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Inspect one prior Run summary from the read-only Research History."""

    _run_summary_command(_resolve_runs_root(runs_root), run_id, json_output)


@app.command("get-run-summary")
def get_run_summary_command(
    run_id: Annotated[str, typer.Option(help="Run identifier to inspect.")],
    runs_root: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing local Harness Run directories. Defaults to /history/runs inside the Agent Control Boundary."
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Alias for run-summary."""

    _run_summary_command(_resolve_runs_root(runs_root), run_id, json_output)


@app.command("get-best-runs")
def get_best_runs_command(
    runs_root: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing local Harness Run directories. Defaults to /history/runs inside the Agent Control Boundary."
        ),
    ] = None,
    metric: Annotated[str, typer.Option(help="Metric key used for ranking local Runs.")] = "val/dice",
    limit: Annotated[int | None, typer.Option(help="Maximum number of ranked Runs to print.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Identify best completed prior Runs in the read-only Research History."""

    rows = get_best_runs(_resolve_runs_root(runs_root), metric=metric, limit=limit)
    if json_output:
        _echo_json(rows)
    else:
        _echo_table(rows)


@app.command("validate-candidate")
def validate_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    workspace_root: Annotated[
        Path,
        typer.Option(help="Research Workspace Root containing ml-autoresearch.toml Research Problem provider config."),
    ] = Path("."),
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
    from ml_autoresearch.research_problems import ResearchProblemProviderLoadError

    try:
        registry = load_configured_research_problem_registry(workspace_root)
        manifest = validate_candidate_directory(
            candidate,
            require_proposal=require_proposal,
            require_readme=require_readme,
            research_problem_registry=registry,
        )
    except (CandidateExecutionConfigError, ResearchProblemProviderLoadError, CandidateValidationError, OSError) as exc:
        _echo_json({"status": "invalid", "reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json({"status": "valid", "manifest": manifest.model_dump(mode="json")})


@app.command("validate-capability-request")
def validate_capability_request_command(
    request: Annotated[Path, typer.Option(help="Path to a YAML Capability Request file.")],
) -> None:
    """Validate a Capability Request handoff without ingesting it or writing the ledger."""

    try:
        validated = validate_capability_request_file(request)
    except CapabilityRequestError as exc:
        _echo_json({"status": "invalid", "reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json({"status": "valid", "request": validated.model_dump(exclude_none=True)})


@app.command("create-capability-request")
def create_capability_request_command(
    output: Annotated[Path, typer.Option(help="Output YAML Capability Request path in the Agent Workspace.")],
    capability_type: Annotated[str, typer.Option(help="Capability type, e.g. contract_surface or dataset_profile_artifact.")],
    blocked_hypothesis: Annotated[str, typer.Option(help="Research hypothesis blocked by the current Harness-owned surface.")],
    current_contract_insufficiency: Annotated[
        str,
        typer.Option(help="Why the current contract, policy, or exposed context is insufficient."),
    ],
    expected_research_value: Annotated[str, typer.Option(help="Learning value unlocked by the request.")],
    safety_reproducibility_risks: Annotated[str, typer.Option(help="Risks for human reviewers.")],
    minimal_harness_change: Annotated[str, typer.Option(help="Smallest Harness-owned change that could unblock the hypothesis.")],
    candidate_authority_requested: Annotated[
        str,
        typer.Option(help="Candidate authority requested; prefer none."),
    ] = "none",
    example_follow_up_experiments: Annotated[
        list[str] | None,
        typer.Option("--example-follow-up-experiment", help="Repeat for each follow-up experiment string."),
    ] = None,
    priority: Annotated[str, typer.Option(help="Priority: low, medium, high, or urgent.")] = "medium",
    request_id: Annotated[str | None, typer.Option(help="Stable request id; defaults to output filename stem.")] = None,
    diagnostic_question: Annotated[str | None, typer.Option(help="Required for dataset_profile_artifact requests.")] = None,
    expected_research_decision_impact: Annotated[
        str | None,
        typer.Option(help="Required for dataset_profile_artifact requests."),
    ] = None,
    scope_split: Annotated[str | None, typer.Option(help="Required for dataset_profile_artifact requests.")] = None,
    bounded_computation_artifact_budget: Annotated[
        str | None,
        typer.Option(help="Required for dataset_profile_artifact requests."),
    ] = None,
    provenance_requirements: Annotated[str | None, typer.Option(help="Required for dataset_profile_artifact requests.")] = None,
) -> None:
    """Create a safe YAML Capability Request from structured CLI fields."""

    data = {
        "request_id": request_id,
        "capability_type": capability_type,
        "blocked_hypothesis": blocked_hypothesis,
        "current_contract_insufficiency": current_contract_insufficiency,
        "expected_research_value": expected_research_value,
        "safety_reproducibility_risks": safety_reproducibility_risks,
        "minimal_harness_change": minimal_harness_change,
        "candidate_authority_requested": candidate_authority_requested,
        "example_follow_up_experiments": example_follow_up_experiments or [],
        "priority": priority,
        "diagnostic_question": diagnostic_question,
        "expected_research_decision_impact": expected_research_decision_impact,
        "scope_split": scope_split,
        "bounded_computation_artifact_budget": bounded_computation_artifact_budget,
        "provenance_requirements": provenance_requirements,
    }
    try:
        written = write_capability_request_file(output, data)
    except CapabilityRequestError as exc:
        _echo_json({"status": "invalid", "reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json({"status": "created", "path": str(output), "request": written.model_dump(exclude_none=True)})


@app.command("prepare-experiment-batch-submission")
def prepare_experiment_batch_submission_command(
    batch: Annotated[Path, typer.Option(help="Path to a draft Experiment Batch directory.")],
    submissions_root: Annotated[Path, typer.Option(help="Root of the immutable Experiment Batch Submission Queue.")],
) -> None:
    """Statically validate and copy a draft Experiment Batch into the submission queue."""

    from ml_autoresearch.submissions import CandidateSubmissionPreparationError, prepare_experiment_batch_submission

    try:
        result = prepare_experiment_batch_submission(batch, submissions_root)
    except (CandidateSubmissionPreparationError, OSError) as exc:
        _echo_json({"status": "rejected", "rejection_reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json(result)


@app.command("prepare-candidate-submission")
def prepare_candidate_submission_command(
    candidate: Annotated[Path, typer.Option(help="Path to a draft Candidate Experiment directory.")],
    submissions_root: Annotated[Path, typer.Option(help="Root of the immutable Candidate Submission Queue.")],
    workspace_root: Annotated[
        Path,
        typer.Option(help="Research Workspace Root containing ml-autoresearch.toml Research Problem provider config."),
    ] = Path("."),
) -> None:
    """Statically validate and copy a draft Candidate Experiment into the submission queue."""

    from ml_autoresearch.candidate_execution_config import CandidateExecutionConfigError, load_configured_research_problem_registry
    from ml_autoresearch.research_problems import ResearchProblemProviderLoadError
    from ml_autoresearch.submissions import CandidateSubmissionPreparationError, prepare_candidate_submission

    try:
        registry = load_configured_research_problem_registry(workspace_root)
        result = prepare_candidate_submission(candidate, submissions_root, research_problem_registry=registry)
    except (CandidateExecutionConfigError, ResearchProblemProviderLoadError, CandidateSubmissionPreparationError, OSError) as exc:
        _echo_json({"status": "rejected", "rejection_reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json(result)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
