"""Execution backends for Candidate Experiment smoke testing."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml_autoresearch.research_problems import ResearchProblemProviderConfig

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

    def train_synthetic(
        self, run_dir: str | Path, *, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n"
    ) -> OperationResult:
        """Train the copied Candidate Experiment on deterministic synthetic fixture data."""

    def train_gvccs(
        self,
        run_dir: str | Path,
        data_root: str | Path,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        """Train the copied Candidate Experiment on Harness-owned GVCCS data loading."""

    def train_research_problem(
        self,
        run_dir: str | Path,
        provider_config: ResearchProblemProviderConfig,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        """Train through the configured trusted Research Problem provider."""

    def evaluate_run(
        self,
        run_dir: str | Path,
        *,
        data_root: str | Path | None = None,
        max_artifact_samples: int = 12,
    ) -> OperationResult:
        """Evaluate a completed Run without retraining."""


@dataclass(frozen=True)
class NativeBackend:
    """Developer-unsafe native process backend."""

    name: str = "native"
    developer_unsafe: bool = True

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        from ml_autoresearch.smoke import smoke_test_run

        result = smoke_test_run(run_dir)
        return OperationResult(
            backend=self.name,
            operation="smoke_test",
            parameter_count=result.parameter_count,
            input_spec=result.input_spec,
            output_spec=result.output_spec,
        )

    def train_synthetic(
        self, run_dir: str | Path, *, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n"
    ) -> OperationResult:
        from ml_autoresearch.training import train_synthetic_fixture_run

        train_synthetic_fixture_run(
            run_dir, max_prediction_samples=max_prediction_samples, prediction_sample_policy=prediction_sample_policy
        )
        return OperationResult(backend=self.name, operation="train_synthetic")

    def train_gvccs(
        self,
        run_dir: str | Path,
        data_root: str | Path,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        from ml_autoresearch.training import train_gvccs_run

        train_gvccs_run(
            run_dir,
            data_root,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        )
        return OperationResult(backend=self.name, operation="train_gvccs")

    def train_research_problem(
        self,
        run_dir: str | Path,
        provider_config: ResearchProblemProviderConfig,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        from ml_autoresearch.training import train_research_problem_run

        train_research_problem_run(
            run_dir,
            provider_config,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        )
        return OperationResult(backend=self.name, operation="train_research_problem")

    def evaluate_run(
        self,
        run_dir: str | Path,
        *,
        data_root: str | Path | None = None,
        max_artifact_samples: int = 12,
    ) -> OperationResult:
        from ml_autoresearch.evaluations import evaluate_run  # local import avoids a module cycle.

        evaluate_run(run_dir, backend="native", data_root=data_root, max_artifact_samples=max_artifact_samples)
        return OperationResult(backend=self.name, operation="evaluate_run")


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

    def train_synthetic(
        self, run_dir: str | Path, *, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n"
    ) -> OperationResult:
        path = Path(run_dir)
        self._prepare_writable_paths(path)
        self._ensure_image_available()
        command = self._operation_command(
            path,
            "ml_autoresearch.container_runner",
            "train-synthetic",
            f"--max-prediction-samples={max_prediction_samples}",
            f"--prediction-sample-policy={prediction_sample_policy}",
        )
        return self._run_training_operation(command, path, "Docker synthetic training failed", "train_synthetic")

    def train_gvccs(
        self,
        run_dir: str | Path,
        data_root: str | Path,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        path = Path(run_dir)
        data_path = self._validate_gvccs_data_root(data_root)
        self._prepare_writable_paths(path)
        self._ensure_image_available()
        args = ["train-gvccs"]
        if max_samples is not None:
            args.append(f"--max-samples={max_samples}")
        args.append(f"--max-prediction-samples={max_prediction_samples}")
        args.append(f"--prediction-sample-policy={prediction_sample_policy}")
        command = self._operation_command(path, "ml_autoresearch.container_runner", *args, data_root=data_path)
        return self._run_training_operation(command, path, "Docker GVCCS training failed", "train_gvccs")

    def train_research_problem(
        self,
        run_dir: str | Path,
        provider_config: ResearchProblemProviderConfig,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        path = Path(run_dir)
        provider_package_root = self._validate_research_problem_package_root(provider_config.package_root)
        data_path = self._research_problem_data_root(provider_config)
        container_config = self._container_research_problem_config(provider_config, data_root_mounted=data_path is not None)
        self._prepare_writable_paths(path)
        self._ensure_image_available()
        args = ["train-research-problem", f"--provider-config-json={container_config.model_dump_json()}"]
        if max_samples is not None:
            args.append(f"--max-samples={max_samples}")
        args.append(f"--max-prediction-samples={max_prediction_samples}")
        args.append(f"--prediction-sample-policy={prediction_sample_policy}")
        command = self._operation_command(
            path,
            "ml_autoresearch.container_runner",
            *args,
            data_root=data_path,
            provider_package_root=provider_package_root,
        )
        return self._run_training_operation(command, path, "Docker Research Problem training failed", "train_research_problem")

    def evaluate_run(
        self,
        run_dir: str | Path,
        *,
        data_root: str | Path | None = None,
        max_artifact_samples: int = 12,
    ) -> OperationResult:
        path = Path(run_dir)
        data_path = self._evaluate_data_root(path, data_root)
        self._prepare_evaluation_writable_paths(path)
        self._ensure_image_available()
        command = self._operation_command(
            path,
            "ml_autoresearch.container_runner",
            "evaluate-run",
            "--data-root=/data",
            f"--max-artifact-samples={max_artifact_samples}",
            "--backend=native",
            data_root=data_path,
            outputs_read_only=True,
            evaluations_writable=True,
        )
        self._run_operation(command, "Docker evaluation failed")
        return OperationResult(backend=self.name, operation="evaluate_run", docker_image=self.docker_image)

    def _prepare_writable_paths(self, path: Path) -> None:
        outputs = path / "outputs"
        logs = outputs / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (path / "scratch").mkdir(parents=True, exist_ok=True)
        if self.container_user is not None and self.container_user != self._host_user() and not self.rootless_container_root:
            os.chmod(outputs, 0o777)
            os.chmod(logs, 0o777)

    def _prepare_evaluation_writable_paths(self, path: Path) -> None:
        evaluations = path / "outputs" / "evaluations"
        evaluations.mkdir(parents=True, exist_ok=True)
        (path / "scratch").mkdir(parents=True, exist_ok=True)
        if self.container_user is not None and self.container_user != self._host_user() and not self.rootless_container_root:
            os.chmod(evaluations, 0o777)

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
        outputs_read_only: bool = False,
        evaluations_writable: bool = False,
        provider_package_root: Path | None = None,
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
            f"{run_dir / 'outputs'}:/outputs:{'ro' if outputs_read_only else 'rw'},z",
            "--mount",
            f"type=tmpfs,destination=/scratch,tmpfs-size={self.scratch_size},tmpfs-mode=1777",
        ]
        if evaluations_writable:
            command.extend(["--volume", f"{run_dir / 'outputs' / 'evaluations'}:/outputs/evaluations:rw,z"])
        if self.enable_gpu:
            command.extend(["--gpus", "all"])
        if data_root is not None:
            command.extend(["--volume", f"{data_root}:/data:ro,z"])
        if provider_package_root is not None:
            command.extend(["--volume", f"{provider_package_root}:/research_problem_package:ro,z"])
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

    def _evaluate_data_root(self, run_dir: Path, data_root: str | Path | None) -> Path:
        if data_root is not None:
            return self._validate_gvccs_data_root(data_root)
        try:
            metadata = json.loads((run_dir / "run_metadata.json").read_text())
        except Exception as exc:  # noqa: BLE001 - Docker launch should fail clearly before importing candidate code.
            raise RuntimeError(f"cannot read Run metadata for evaluation data root: {exc}") from exc
        dataset = metadata.get("dataset")
        if not isinstance(dataset, dict) or not isinstance(dataset.get("host_data_path"), str):
            raise RuntimeError("Run metadata does not contain GVCCS data root; pass --data-root")
        return self._validate_gvccs_data_root(str(dataset["host_data_path"]))

    def _validate_gvccs_data_root(self, data_root: str | Path) -> Path:
        return self._validate_host_directory(data_root, label="GVCCS data root")

    def _validate_research_problem_package_root(self, package_root: str | Path) -> Path:
        return self._validate_host_directory(package_root, label="Research Problem package root")

    def _research_problem_data_root(self, provider_config: ResearchProblemProviderConfig) -> Path | None:
        data_root = provider_config.data_config.get("dataset_root") or provider_config.data_config.get("data_root")
        if data_root is None:
            return None
        if not isinstance(data_root, str):
            raise RuntimeError("Research Problem data root must be configured as a string")
        return self._validate_host_directory(data_root, label="Research Problem data root")

    def _container_research_problem_config(
        self,
        provider_config: ResearchProblemProviderConfig,
        *,
        data_root_mounted: bool,
    ) -> ResearchProblemProviderConfig:
        data_config = dict(provider_config.data_config)
        if data_root_mounted:
            if "dataset_root" in data_config:
                data_config["dataset_root"] = "/data"
            if "data_root" in data_config:
                data_config["data_root"] = "/data"
        return provider_config.model_copy(
            update={"package_root": Path("/research_problem_package"), "data_config": data_config}
        )

    def _validate_host_directory(self, value: str | Path, *, label: str) -> Path:
        path = Path(value)
        if not path.exists():
            raise RuntimeError(f"{label} does not exist: {path}")
        if not path.is_dir():
            raise RuntimeError(f"{label} is not a directory: {path}")
        try:
            return path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"{label} cannot be resolved: {path}: {exc}") from exc


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
