"""Execution backends for Candidate Experiment smoke testing."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml_autoresearch.smoke import SmokeTestResult, smoke_test_run

DEFAULT_DOCKER_IMAGE = "ml-autoresearch-runner:local"
MANUAL_DOCKER_BUILD_COMMAND = f"docker build -t {DEFAULT_DOCKER_IMAGE} ."


@dataclass(frozen=True)
class OperationResult:
    """Result from an execution backend operation."""

    backend: str
    parameter_count: int | None = None
    input_spec: dict[str, object] | None = None
    output_spec: dict[str, object] | None = None
    docker_image: str | None = None


class ExecutionBackend(Protocol):
    """Boundary used by the Harness to execute Candidate Experiment operations."""

    name: str

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        """Smoke-test the copied Candidate Experiment for a Run."""


@dataclass(frozen=True)
class NativeBackend:
    """Developer-unsafe native process backend."""

    name: str = "native"
    developer_unsafe: bool = True

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        result: SmokeTestResult = smoke_test_run(run_dir)
        return OperationResult(
            backend=self.name,
            parameter_count=result.parameter_count,
            input_spec=result.input_spec,
            output_spec=result.output_spec,
        )


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
        command = self._smoke_test_command(path)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"Docker smoke test failed: {detail}") from exc
        return OperationResult(backend=self.name, docker_image=self.docker_image)

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

    def _smoke_test_command(self, run_dir: Path) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--mount",
            f"type=bind,src={run_dir / 'candidate'},dst=/candidate,readonly",
            "--mount",
            f"type=bind,src={run_dir / 'resolved_manifest.yaml'},dst=/resolved_manifest.yaml,readonly",
            "--mount",
            f"type=bind,src={run_dir / 'run_metadata.json'},dst=/run_metadata.json,readonly",
            "--mount",
            f"type=bind,src={run_dir / 'outputs'},dst=/outputs",
            "--mount",
            f"type=bind,src={run_dir / 'scratch'},dst=/scratch",
            "--entrypoint",
            "python",
            self.docker_image,
            "-m",
            "ml_autoresearch.container_smoke",
        ]


def backend_metadata(backend: ExecutionBackend) -> dict[str, object]:
    if isinstance(backend, DockerBackend):
        return {"name": backend.name, "docker_image": backend.docker_image}
    if isinstance(backend, NativeBackend):
        return {"name": backend.name, "developer_unsafe": True}
    return {"name": backend.name}
