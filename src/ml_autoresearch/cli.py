"""Command-line interface for ML Autoresearch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ml_autoresearch.runs import RunStatus, submit_candidate

app = typer.Typer(help="ML Autoresearch local Harness commands.")


@app.callback()
def root() -> None:
    """ML Autoresearch local Harness commands."""


@app.command("submit-candidate")
def submit_candidate_command(
    candidate: Annotated[Path, typer.Option(help="Path to a local Candidate Experiment directory.")],
    runs_root: Annotated[Path, typer.Option(help="Directory where Harness Run directories are created.")],
) -> None:
    """Validate a local Candidate Experiment and create a Run."""

    run = submit_candidate(candidate, runs_root)
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
    if run.status in {RunStatus.REJECTED, RunStatus.SMOKE_FAILED}:
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
