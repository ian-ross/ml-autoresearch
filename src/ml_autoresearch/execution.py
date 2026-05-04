"""Execution backends for Candidate Experiment smoke testing."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml_autoresearch.smoke import SmokeTestResult, smoke_test_run
from ml_autoresearch.training import train_gvccs_run, train_synthetic_fixture_run

DEFAULT_DOCKER_IMAGE = "ml-autoresearch-runner:local"
MANUAL_DOCKER_BUILD_COMMAND = f"docker build -t {DEFAULT_DOCKER_IMAGE} ."
DEFAULT_DOCKER_MEMORY_LIMIT = "4g"
DEFAULT_DOCKER_CPUS = "2"
DEFAULT_DOCKER_PIDS_LIMIT = "512"
DEFAULT_DOCKER_SCRATCH_SIZE = "2g"
DEFAULT_TIMEOUT_GRACE_SECONDS = 30
DOCKER_GPU_VALIDATION_PROBE = """
import json
import torch

payload = {
    "torch.__version__": torch.__version__,
    "torch.version.cuda": torch.version.cuda,
    "torch.cuda.is_available()": torch.cuda.is_available(),
    "torch.cuda.device_count()": torch.cuda.device_count(),
    "driver_visible_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
""".strip()


class DockerOperationTimeoutError(RuntimeError):
    """Raised when Docker training exceeds the wall-clock budget and grace period."""

    def __init__(self, message: str, *, timeout_metadata: dict[str, object]):
        super().__init__(message)
        self.timeout_metadata = timeout_metadata


@dataclass(frozen=True)
class OperationResult:
    """Result from an execution backend operation."""

    backend: str
    operation: str
    parameter_count: int | None = None
    input_spec: dict[str, object] | None = None
    output_spec: dict[str, object] | None = None
    docker_image: str | None = None
    lifecycle_status: str = "completed"
    timeout: dict[str, object] | None = None


class ExecutionBackend(Protocol):
    """Boundary used by the Harness to execute Candidate Experiment operations."""

    name: str

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        """Smoke-test the copied Candidate Experiment for a Run."""

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2) -> OperationResult:
        """Train the copied Candidate Experiment on deterministic synthetic fixture data."""

    def train_gvccs(
        self,
        run_dir: str | Path,
        data_root: str | Path,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
    ) -> OperationResult:
        """Train the copied Candidate Experiment on Harness-owned GVCCS data loading."""


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

    def train_gvccs(
        self,
        run_dir: str | Path,
        data_root: str | Path,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
    ) -> OperationResult:
        train_gvccs_run(run_dir, data_root, max_samples=max_samples, max_prediction_samples=max_prediction_samples)
        return OperationResult(backend=self.name, operation="train_gvccs")


@dataclass(frozen=True)
class DockerBackend:
    """Docker-backed Candidate Execution Boundary."""

    docker_image: str = DEFAULT_DOCKER_IMAGE
    name: str = "docker"
    wall_clock_timeout_seconds: float | None = None
    timeout_grace_seconds: int = DEFAULT_TIMEOUT_GRACE_SECONDS
    memory_limit: str = DEFAULT_DOCKER_MEMORY_LIMIT
    cpus: str = DEFAULT_DOCKER_CPUS
    pids_limit: str = DEFAULT_DOCKER_PIDS_LIMIT
    scratch_size: str = DEFAULT_DOCKER_SCRATCH_SIZE
    enable_gpu: bool = False
    container_user: str | None = None
    rootless_container_root: bool = False

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        path = Path(run_dir)
        self._prepare_writable_paths(path)
        self._ensure_image_available()
        command = self._operation_command(path, "ml_autoresearch.container_smoke")
        self._run_operation(command, "Docker smoke test failed")
        return OperationResult(backend=self.name, operation="smoke_test", docker_image=self.docker_image)

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2) -> OperationResult:
        path = Path(run_dir)
        self._prepare_writable_paths(path)
        self._ensure_image_available()
        command = self._operation_command(
            path,
            "ml_autoresearch.container_runner",
            "train-synthetic",
            f"--max-prediction-samples={max_prediction_samples}",
        )
        return self._run_training_operation(command, path, "Docker synthetic training failed", "train_synthetic")

    def train_gvccs(
        self,
        run_dir: str | Path,
        data_root: str | Path,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
    ) -> OperationResult:
        path = Path(run_dir)
        data_path = self._validate_gvccs_data_root(data_root)
        self._prepare_writable_paths(path)
        self._ensure_image_available()
        args = ["train-gvccs"]
        if max_samples is not None:
            args.append(f"--max-samples={max_samples}")
        args.append(f"--max-prediction-samples={max_prediction_samples}")
        command = self._operation_command(path, "ml_autoresearch.container_runner", *args, data_root=data_path)
        return self._run_training_operation(command, path, "Docker GVCCS training failed", "train_gvccs")

    def _prepare_writable_paths(self, path: Path) -> None:
        outputs = path / "outputs"
        logs = outputs / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (path / "scratch").mkdir(parents=True, exist_ok=True)
        if self.container_user is not None and self.container_user != self._host_user() and not self.rootless_container_root:
            os.chmod(outputs, 0o777)
            os.chmod(logs, 0o777)

    def _run_training_operation(
        self, command: list[str], run_dir: Path, failure_prefix: str, operation: str
    ) -> OperationResult:
        if self.wall_clock_timeout_seconds is None:
            self._run_operation(command, failure_prefix)
            return OperationResult(backend=self.name, operation=operation, docker_image=self.docker_image)
        timeout_metadata = self._run_operation_with_graceful_timeout(command, run_dir, failure_prefix)
        return OperationResult(
            backend=self.name,
            operation=operation,
            docker_image=self.docker_image,
            lifecycle_status="timeout_graceful" if timeout_metadata else "completed",
            timeout=timeout_metadata,
        )

    def _run_operation(self, command: list[str], failure_prefix: str) -> None:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr or exc.stdout or str(exc)
            raise RuntimeError(f"{failure_prefix}: {detail}") from exc

    def _run_operation_with_graceful_timeout(
        self, command: list[str], run_dir: Path, failure_prefix: str
    ) -> dict[str, object] | None:
        container_name = _docker_arg_value(command, "--name")
        assert container_name is not None
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=self.wall_clock_timeout_seconds)
        except subprocess.TimeoutExpired:
            _signal_container_timeout(container_name)
            _append_harness_timeout_log(
                run_dir,
                f"Wall-clock budget exhausted after {self.wall_clock_timeout_seconds}s; grace period {self.timeout_grace_seconds}s started.",
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_grace_seconds)
            except subprocess.TimeoutExpired as exc:
                subprocess.run(["docker", "kill", container_name], check=False, capture_output=True, text=True)
                stdout, stderr = proc.communicate()
                metadata = {
                    "requested": True,
                    "wall_clock_timeout_seconds": self.wall_clock_timeout_seconds,
                    "grace_seconds": self.timeout_grace_seconds,
                    "forced_termination": True,
                }
                _append_harness_timeout_log(run_dir, "Grace period expired; container was force-terminated.")
                detail = stderr or stdout or str(exc)
                raise DockerOperationTimeoutError(
                    f"{failure_prefix}: wall-clock budget exhausted and grace period expired: {detail}",
                    timeout_metadata=metadata,
                ) from exc
            metadata = {
                "requested": True,
                "wall_clock_timeout_seconds": self.wall_clock_timeout_seconds,
                "grace_seconds": self.timeout_grace_seconds,
                "forced_termination": False,
            }
            if proc.returncode != 0:
                detail = stderr or stdout or f"container exited with status {proc.returncode}"
                raise RuntimeError(f"{failure_prefix}: {detail}")
            _append_harness_timeout_log(run_dir, "Container exited cleanly during timeout grace period.")
            return metadata
        if proc.returncode != 0:
            detail = stderr or stdout or f"container exited with status {proc.returncode}"
            raise RuntimeError(f"{failure_prefix}: {detail}")
        return None

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

    def _operation_command(
        self,
        run_dir: Path,
        module: str,
        *args: str,
        data_root: Path | None = None,
    ) -> list[str]:
        container_name = f"ml-autoresearch-{run_dir.name}-{uuid.uuid4().hex[:12]}"
        docker_is_rootless = self._docker_is_rootless() if self.container_user is None and not self.rootless_container_root else False
        command = [
            "docker",
            "run",
            "--rm",
            *([] if docker_is_rootless or self.rootless_container_root or self.container_user is not None else ["--userns=host"]),
            "--name",
            container_name,
            "--network",
            "none",
            "--user",
            self._container_user(docker_is_rootless=docker_is_rootless),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            self.pids_limit,
            "--memory",
            self.memory_limit,
            "--cpus",
            self.cpus,
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
            "--env",
            "ML_AUTORESEARCH_TIMEOUT_SENTINEL=/scratch/ml_autoresearch_timeout_requested",
            "--volume",
            f"{run_dir / 'candidate'}:/candidate:ro,z",
            "--volume",
            f"{run_dir / 'resolved_manifest.yaml'}:/resolved_manifest.yaml:ro,z",
            "--volume",
            f"{run_dir / 'run_metadata.json'}:/run_metadata.json:ro,z",
            "--volume",
            f"{run_dir / 'outputs'}:/outputs:rw,z",
            "--mount",
            f"type=tmpfs,destination=/scratch,tmpfs-size={self.scratch_size},tmpfs-mode=1777",
        ]
        if self.enable_gpu:
            command.extend(["--gpus", "all"])
        if data_root is not None:
            command.extend(["--volume", f"{data_root}:/data:ro,z"])
        command.extend(["--entrypoint", "python", self.docker_image, "-m", module, *args])
        return command

    def _container_user(self, *, docker_is_rootless: bool = False) -> str:
        if self.rootless_container_root or (docker_is_rootless and self.container_user is None):
            return "0:0"
        return self.container_user or self._host_user()

    @staticmethod
    def _docker_is_rootless() -> bool:
        try:
            completed = subprocess.run(
                ["docker", "info", "--format", "{{json .SecurityOptions}}"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            return False
        try:
            security_options = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return False
        return any("rootless" in str(option) for option in security_options)

    @staticmethod
    def _host_user() -> str:
        return f"{os.getuid()}:{os.getgid()}"

    def _validate_gvccs_data_root(self, data_root: str | Path) -> Path:
        path = Path(data_root)
        if not path.exists():
            raise RuntimeError(f"GVCCS data root does not exist: {path}")
        if not path.is_dir():
            raise RuntimeError(f"GVCCS data root is not a directory: {path}")
        try:
            return path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"GVCCS data root cannot be resolved: {path}: {exc}") from exc


def docker_gpu_validation_command(docker_image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    """Build the safe runner-image GPU validation command."""

    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "none",
        "--entrypoint",
        "python",
        docker_image,
        "-c",
        DOCKER_GPU_VALIDATION_PROBE,
    ]


def validate_docker_gpu(docker_image: str = DEFAULT_DOCKER_IMAGE) -> subprocess.CompletedProcess[str]:
    """Run the runner-image GPU validation probe without launching candidate code."""

    return subprocess.run(
        docker_gpu_validation_command(docker_image),
        check=False,
        capture_output=True,
        text=True,
    )


def _docker_arg_value(command: list[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _signal_container_timeout(container_name: str) -> None:
    subprocess.run(
        [
            "docker",
            "exec",
            container_name,
            "python",
            "-c",
            "from pathlib import Path; Path('/scratch/ml_autoresearch_timeout_requested').write_text('timeout_requested\n')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _append_harness_timeout_log(run_dir: Path, line: str) -> None:
    log_path = run_dir / "outputs" / "logs" / "harness_timeout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(line + "\n")


def backend_metadata(backend: ExecutionBackend) -> dict[str, object]:
    if isinstance(backend, DockerBackend):
        docker_is_rootless = (
            backend._docker_is_rootless() if backend.container_user is None and not backend.rootless_container_root else False
        )
        rootless_container_root = backend.rootless_container_root or (docker_is_rootless and backend.container_user is None)
        metadata: dict[str, object] = {
            "name": backend.name,
            "docker_image": backend.docker_image,
            "gpu_policy": "disabled_by_default" if not backend.enable_gpu else "enabled_by_harness_configuration",
            "docker_user": backend._container_user(docker_is_rootless=docker_is_rootless),
            "rootless_container_root": rootless_container_root,
            "user_namespace": "rootless" if docker_is_rootless else "host",
            "resource_limits": {
                "memory": backend.memory_limit,
                "cpus": backend.cpus,
                "pids_limit": backend.pids_limit,
                "scratch": backend.scratch_size,
            },
        }
        if backend.wall_clock_timeout_seconds is not None:
            metadata["wall_clock_budget"] = {
                "timeout_seconds": backend.wall_clock_timeout_seconds,
                "grace_seconds": backend.timeout_grace_seconds,
            }
        return metadata
    if isinstance(backend, NativeBackend):
        return {"name": backend.name, "developer_unsafe": True}
    return {"name": backend.name}
