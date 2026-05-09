"""Opt-in Docker integration tests for the Candidate Execution Boundary.

These tests intentionally run a real Docker build and container. They are skipped
by default because they require Docker daemon access and take longer than unit
suite command-construction tests.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


RUN_DOCKER_INTEGRATION = os.environ.get("ML_AUTORESEARCH_DOCKER_INTEGRATION") == "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_IMAGE = "ml-autoresearch-runner:integration-test"


def _run(command: list[str], *, cwd: Path = PROJECT_ROOT, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_DOCKER_INTEGRATION,
    reason="set ML_AUTORESEARCH_DOCKER_INTEGRATION=1 to run real Docker integration tests",
)
def test_docker_synthetic_research_loop_runs_in_real_container(tmp_path: Path):
    """Build the runner image and run the full synthetic fixture loop in Docker."""

    if shutil.which("docker") is None:
        pytest.fail("Docker executable is not available")

    build = _run(["docker", "build", "-t", INTEGRATION_IMAGE, "."], timeout=600)
    assert build.returncode == 0, f"docker build failed\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"

    candidate = PROJECT_ROOT / "tests" / "fixtures" / "candidates" / "single_frame_unet_baseline"
    runs_root = tmp_path / "runs"
    completed = _run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "run-candidate",
            "--candidate",
            str(candidate),
            "--runs-root",
            str(runs_root),
            "--synthetic-fixture",
            "--docker-image",
            INTEGRATION_IMAGE,
            "--no-require-proposal",
        ],
        timeout=600,
    )
    assert completed.returncode == 0, (
        "Docker synthetic Research Loop failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    run_dir = Path(payload["run_dir"])

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["execution_backend"]["name"] == "docker"
    assert metadata["execution_backend"]["docker_image"] == INTEGRATION_IMAGE

    assert (run_dir / "outputs" / "model_summary.json").is_file()
    assert (run_dir / "outputs" / "metrics.jsonl").is_file()
    final_metrics_path = run_dir / "outputs" / "final_metrics.json"
    assert final_metrics_path.is_file()
    assert (run_dir / "outputs" / "logs" / "smoke_test.log").is_file()
    assert (run_dir / "outputs" / "logs" / "training.log").is_file()

    final_metrics = json.loads(final_metrics_path.read_text())
    assert "val/dice" in final_metrics
    prediction_samples = final_metrics["artifacts"]["prediction_samples"]
    assert (run_dir / prediction_samples).is_file()
    sample_manifest = json.loads((run_dir / prediction_samples).read_text())
    assert sample_manifest["status"] == "completed"
    assert sample_manifest["sample_count"] > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_DOCKER_INTEGRATION,
    reason="set ML_AUTORESEARCH_DOCKER_INTEGRATION=1 to run real Docker integration tests",
)
def test_docker_gvccs_like_fixture_training_runs_in_real_container(tmp_path: Path):
    """Build the runner image and train on the GVCCS-like fixture through read-only /data."""

    if shutil.which("docker") is None:
        pytest.fail("Docker executable is not available")

    build = _run(["docker", "build", "-t", INTEGRATION_IMAGE, "."], timeout=600)
    assert build.returncode == 0, f"docker build failed\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}"

    candidate = PROJECT_ROOT / "tests" / "fixtures" / "candidates" / "single_frame_unet_baseline"
    data_root = PROJECT_ROOT / "tests" / "fixtures" / "gvccs_like"
    runs_root = tmp_path / "runs"
    completed = _run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "run-candidate",
            "--candidate",
            str(candidate),
            "--runs-root",
            str(runs_root),
            "--data-root",
            str(data_root),
            "--max-samples",
            "4",
            "--docker-image",
            INTEGRATION_IMAGE,
            "--no-require-proposal",
        ],
        timeout=600,
    )
    assert completed.returncode == 0, (
        "Docker GVCCS fixture Research Loop failed\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    run_dir = Path(payload["run_dir"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["dataset"] == {
        "id": "gvccs",
        "host_data_path": str(data_root.resolve()),
        "container_data_path": "/data",
    }
    assert "Starting GVCCS training from /data" in (run_dir / "outputs" / "logs" / "training.log").read_text()
    assert (run_dir / "outputs" / "final_metrics.json").is_file()
