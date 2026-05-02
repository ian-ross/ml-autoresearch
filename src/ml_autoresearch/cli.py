"""Command-line interface for ML Autoresearch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ml_autoresearch.runs import RunStatus, run_candidate_with_gvccs_data, run_candidate_with_synthetic_fixture, submit_candidate

app = typer.Typer(help="ML Autoresearch local Harness commands.")


@app.callback()
def root() -> None:
    """ML Autoresearch local Harness commands."""


def _echo_run(run) -> None:
    typer.echo(
        json.dumps(
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "status": run.status.value,
                "rejection_reason": run.rejection_reason,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("submit-candidate")
def submit_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    runs_root: Annotated[Path, typer.Option(help="Directory where Harness Run directories are created.")],
) -> None:
    """Validate a local Candidate Experiment and create a Run."""

    run = submit_candidate(candidate, runs_root)
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
) -> None:
    """Validate, smoke-test, and synchronously run a Candidate Experiment."""

    if synthetic_fixture and data_root is not None:
        raise typer.BadParameter("choose either --synthetic-fixture or --data-root, not both")
    if synthetic_fixture:
        run = run_candidate_with_synthetic_fixture(candidate, runs_root)
    elif data_root is not None:
        run = run_candidate_with_gvccs_data(candidate, runs_root, data_root, max_samples=max_samples)
    else:
        raise typer.BadParameter("provide --data-root /path/to/gvccs or --synthetic-fixture")
    _echo_run(run)
    if run.status != RunStatus.COMPLETED:
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
