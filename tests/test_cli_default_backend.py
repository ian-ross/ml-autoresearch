import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


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
    assert metadata["execution_backend"] == {"name": "docker", "docker_image": "ml-autoresearch-runner:local"}


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
