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

from ml_autoresearch.evaluations import DEFAULT_MAX_ARTIFACT_SAMPLES, EvaluationError, evaluate_run
from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, DockerBackend, ExecutionBackend, NativeBackend, validate_docker_gpu
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
        }
    )


def _daemonize_current_run_candidate(runs_root: Path) -> None:
    """Re-exec the current run-candidate command in a detached child process."""

    daemon_logs = runs_root / "daemon_logs"
    daemon_logs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = daemon_logs / f"run_candidate_{timestamp}.log"
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
) -> None:
    """Validate a local Candidate Experiment and create a Run."""

    run = submit_candidate(
        candidate,
        runs_root,
        backend=_select_backend(backend, docker_image, docker_enable_gpu, docker_user, docker_rootless_container_root),
    )
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
    daemonize: Annotated[
        bool,
        typer.Option("--daemonize", help="Start the Candidate Experiment Run in a detached background process and return immediately."),
    ] = False,
) -> None:
    """Validate, smoke-test, and synchronously run a Candidate Experiment."""

    if synthetic_fixture and data_root is not None:
        raise typer.BadParameter("choose either --synthetic-fixture or --data-root, not both")
    if daemonize:
        _daemonize_current_run_candidate(runs_root)
        return
    selected_backend = _select_backend(backend, docker_image, docker_enable_gpu, docker_user, docker_rootless_container_root)
    if synthetic_fixture:
        run = run_candidate_with_synthetic_fixture(
            candidate,
            runs_root,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
            backend=selected_backend,
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
        )
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


@app.command("evaluate-run")
def evaluate_run_command(
    run: Annotated[Path, typer.Option("--run", help="Path to a completed source Run directory.")],
    split: Annotated[Literal["val"], typer.Option("--split", help="Run split to evaluate.")] = "val",
    backend: Annotated[Literal["native"], typer.Option("--backend", help="Post-Run Evaluation backend.")] = "native",
    data_root: Annotated[Path | None, typer.Option("--data-root", help="Override GVCCS Dataset root from Run metadata.")] = None,
    max_artifact_samples: Annotated[
        int,
        typer.Option("--max-artifact-samples", help="Maximum selected diagnostic samples to write as visual artifacts."),
    ] = DEFAULT_MAX_ARTIFACT_SAMPLES,
) -> None:
    """Evaluate a completed Run without retraining and write run-scoped artifacts."""

    try:
        result = evaluate_run(run, split=split, backend=backend, data_root=data_root, max_artifact_samples=max_artifact_samples)
    except EvaluationError as exc:
        _echo_json({"status": "failed", "failure_reason": str(exc)})
        raise typer.Exit(1) from exc
    _echo_json({"evaluation_id": result.evaluation_id, "evaluation_dir": str(result.evaluation_dir), "status": result.status})


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
