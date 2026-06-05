import json
from pathlib import Path

from ml_autoresearch.cli import app
from conftest import invoke_typer_cli


def run_cli(*args: str):
    return invoke_typer_cli(app, args)


def test_run_candidate_defaults_to_docker_backend(tmp_path: Path):
    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(tmp_path / "missing"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--synthetic-fixture",
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    metadata = json.loads((tmp_path / "runs" / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"]["name"] == "docker"
    assert metadata["execution_backend"]["docker_image"] == "ml-autoresearch-runner:local"
    assert metadata["execution_backend"]["gpu_policy"] == "disabled_by_default"


def test_run_candidate_native_escape_hatch_records_developer_unsafe_backend(tmp_path: Path):
    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(tmp_path / "missing"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--synthetic-fixture",
        "--backend",
        "native",
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    metadata = json.loads((tmp_path / "runs" / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"] == {"name": "native", "developer_unsafe": True}
