"""Execution backends for Candidate Experiment smoke testing."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml_autoresearch.smoke import SmokeTestResult, smoke_test_run
from ml_autoresearch.training import train_synthetic_fixture_run

DEFAULT_DOCKER_IMAGE = "ml-autoresearch-runner:local"
MANUAL_DOCKER_BUILD_COMMAND = f"docker build -t {DEFAULT_DOCKER_IMAGE} ."


@dataclass(frozen=True)
class OperationResult:
    """Result from an execution backend operation."""

    backend: str
    operation: str
    parameter_count: int | None = None
    input_spec: dict[str, object] | None = None
    output_spec: dict[str, object] | None = None
    docker_image: str | None = None


class ExecutionBackend(Protocol):
    """Boundary used by the Harness to execute Candidate Experiment operations."""

    name: str

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        """Smoke-test the copied Candidate Experiment for a Run."""

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2) -> OperationResult:
        """Train the copied Candidate Experiment on deterministic synthetic fixture data."""


@dataclass(frozen=True)
class NativeBackend:
    """Developer-unsafe native process backend."""

    name: str = "native"
    developer_unsafe: bool = True

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        result: SmokeTestResult = smoke_test_run(run_dir)
        return OperationResult(
            backend=self.name,
            operation="smoke_test",
            parameter_count=result.parameter_count,
            input_spec=result.input_spec,
            output_spec=result.output_spec,
        )

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2) -> OperationResult:
        train_synthetic_fixture_run(run_dir, max_prediction_samples=max_prediction_samples)
        return OperationResult(backend=self.name, operation="train_synthetic")


@dataclass(frozen=True)
class DockerBackend:
    """Docker-backed Candidate Execution Boundary for smoke tests."""

    docker_image: str = DEFAULT_DOCKER_IMAGE
    name: str = "docker"

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        path = Path(run_dir)
        (path / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
        (path / "scratch").mkdir(parents=True, exist_ok=True)
        self._ensure_image_available()
        command = self._operation_command(path, "ml_autoresearch.container_smoke")
        self._run_operation(command, "Docker smoke test failed")
        return OperationResult(backend=self.name, operation="smoke_test", docker_image=self.docker_image)

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2) -> OperationResult:
        path = Path(run_dir)
        (path / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
        (path / "scratch").mkdir(parents=True, exist_ok=True)
        self._ensure_image_available()
        command = self._operation_command(
            path,
            "ml_autoresearch.container_runner",
            "train-synthetic",
            f"--max-prediction-samples={max_prediction_samples}",
        )
        self._run_operation(command, "Docker synthetic training failed")
        return OperationResult(backend=self.name, operation="train_synthetic", docker_image=self.docker_image)

    def _run_operation(self, command: list[str], failure_prefix: str) -> None:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"{failure_prefix}: {detail}") from exc

    def _ensure_image_available(self) -> None:
        try:
            subprocess.run(["docker", "image", "inspect", self.docker_image], check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Docker executable not found; install Docker and build the local runner image with: "
                               f"{MANUAL_DOCKER_BUILD_COMMAND}") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Docker image '{self.docker_image}' is not available. Build it with: {MANUAL_DOCKER_BUILD_COMMAND}"
            ) from exc

    def _operation_command(self, run_dir: Path, module: str, *args: str) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--read-only",
            "--env",
            "TMPDIR=/scratch",
            "--env",
            "TEMP=/scratch",
            "--env",
            "TMP=/scratch",
            "--env",
            "XDG_CACHE_HOME=/scratch/.cache",
            "--env",
            "TORCHINDUCTOR_CACHE_DIR=/scratch/torchinductor",
            "--volume",
            f"{run_dir / 'candidate'}:/candidate:ro,z",
            "--volume",
            f"{run_dir / 'resolved_manifest.yaml'}:/resolved_manifest.yaml:ro,z",
            "--volume",
            f"{run_dir / 'run_metadata.json'}:/run_metadata.json:ro,z",
            "--volume",
            f"{run_dir / 'outputs'}:/outputs:z",
            "--volume",
            f"{run_dir / 'scratch'}:/scratch:z",
            "--entrypoint",
            "python",
            self.docker_image,
            "-m",
            module,
            *args,
        ]


def backend_metadata(backend: ExecutionBackend) -> dict[str, object]:
    if isinstance(backend, DockerBackend):
        return {"name": backend.name, "docker_image": backend.docker_image}
    if isinstance(backend, NativeBackend):
        return {"name": backend.name, "developer_unsafe": True}
    return {"name": backend.name}
