"""Command-line interface for ML Autoresearch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, DockerBackend, ExecutionBackend, NativeBackend
from ml_autoresearch.runs import (
    RunStatus,
    get_best_runs,
    get_run_summary,
    list_runs,
    run_candidate_with_gvccs_data,
    run_candidate_with_synthetic_fixture,
    submit_candidate,
)

app = typer.Typer(help="ML Autoresearch local Harness commands.")


@app.callback()
def root() -> None:
    """ML Autoresearch local Harness commands."""


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _select_backend(name: str, docker_image: str) -> ExecutionBackend:
    if name == "native":
        return NativeBackend()
    if name == "docker":
        return DockerBackend(docker_image)
    raise typer.BadParameter("backend must be native or docker")


def _echo_run(run) -> None:
    _echo_json(
        {
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "status": run.status.value,
            "rejection_reason": run.rejection_reason,
        }
    )


@app.command("submit-candidate")
def submit_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    runs_root: Annotated[Path, typer.Option(help="Directory where Harness Run directories are created.")],
    backend: Annotated[Literal["native", "docker"], typer.Option("--backend", help="Candidate Execution Boundary backend.")] = "native",
    docker_image: Annotated[str, typer.Option("--docker-image", help="Docker runner image for --backend docker.")] = DEFAULT_DOCKER_IMAGE,
) -> None:
    """Validate a local Candidate Experiment and create a Run."""

    run = submit_candidate(candidate, runs_root, backend=_select_backend(backend, docker_image))
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
    backend: Annotated[Literal["native", "docker"], typer.Option("--backend", help="Candidate Execution Boundary backend.")] = "native",
    docker_image: Annotated[str, typer.Option("--docker-image", help="Docker runner image for --backend docker.")] = DEFAULT_DOCKER_IMAGE,
) -> None:
    """Validate, smoke-test, and synchronously run a Candidate Experiment."""

    if synthetic_fixture and data_root is not None:
        raise typer.BadParameter("choose either --synthetic-fixture or --data-root, not both")
    selected_backend = _select_backend(backend, docker_image)
    if synthetic_fixture:
        run = run_candidate_with_synthetic_fixture(candidate, runs_root, backend=selected_backend)
    elif data_root is not None:
        run = run_candidate_with_gvccs_data(candidate, runs_root, data_root, max_samples=max_samples, backend=selected_backend)
    else:
        raise typer.BadParameter("provide --data-root /path/to/gvccs or --synthetic-fixture")
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
        typer.echo(f"{row.get('run_id', '')}\t{row.get('status', '')}\t{dice}\t{row.get('reason', row.get('error', ''))}")


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
