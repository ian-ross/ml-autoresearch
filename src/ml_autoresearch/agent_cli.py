"""Agent-safe command-line interface for ML Autoresearch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.runs import get_best_runs, get_run_summary, list_runs

app = typer.Typer(
    help=(
        "Agent-safe ML Autoresearch observation and static-preparation commands. "
        "This wrapper cannot run Candidate Experiments."
    )
)


@app.callback()
def root() -> None:
    """Agent-safe commands only; cannot run Candidate Experiments."""


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


@app.command("list-runs")
def list_runs_command(
    runs_root: Annotated[Path, typer.Option(help="Directory containing local Harness Run directories.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """List prior local Runs from a read-only runs/ artifact tree."""

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
    """Inspect one prior local Run summary without MLflow or model imports."""

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
    """Identify best completed prior local Runs by val/dice by default."""

    rows = get_best_runs(runs_root, metric=metric, limit=limit)
    if json_output:
        _echo_json(rows)
    else:
        _echo_table(rows)


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

    try:
        manifest = validate_candidate_directory(candidate, require_proposal=require_proposal, require_readme=require_readme)
    except (CandidateValidationError, OSError) as exc:
        _echo_json({"status": "invalid", "reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json({"status": "valid", "manifest": manifest.model_dump(mode="json")})


@app.command("prepare-candidate-submission")
def prepare_candidate_submission_command(
    candidate: Annotated[Path, typer.Option(help="Path to a draft Candidate Experiment directory.")],
    submissions_root: Annotated[Path, typer.Option(help="Root of the immutable Candidate Submission Queue.")],
) -> None:
    """Statically validate and copy a draft Candidate Experiment into the submission queue."""

    from ml_autoresearch.submissions import CandidateSubmissionPreparationError, prepare_candidate_submission

    try:
        result = prepare_candidate_submission(candidate, submissions_root)
    except (CandidateSubmissionPreparationError, OSError) as exc:
        _echo_json({"status": "rejected", "rejection_reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json(result)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
