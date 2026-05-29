"""Project-local Candidate Execution Boundary configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, ExecutionBackend

CONFIG_FILENAME = "candidate-execution.toml"


class CandidateExecutionConfigError(ValueError):
    """Raised when Candidate Execution Boundary configuration is invalid."""


@dataclass(frozen=True)
class CandidateExecutionConfig:
    """Harness-owned execution policy for Candidate Experiment Runs."""

    backend: Literal["native", "docker"] = "native"
    docker_image: str = DEFAULT_DOCKER_IMAGE
    docker_enable_gpu: bool = False
    docker_user: str | None = None
    docker_rootless_container_root: bool = False
    data_root: Path | None = None
    max_samples: int | None = None
    max_prediction_samples: int = 2
    prediction_sample_policy: Literal["first_n", "adjacent_and_scattered"] = "first_n"


def load_candidate_execution_config(project_root: str | Path = Path(".")) -> CandidateExecutionConfig:
    """Load Candidate Execution Boundary config from candidate-execution.toml if present."""

    root = Path(project_root).resolve()
    path = root / CONFIG_FILENAME
    if not path.is_file():
        return CandidateExecutionConfig()
    data = tomllib.loads(path.read_text())
    settings = data.get("candidate_execution", {})
    if not isinstance(settings, dict):
        raise CandidateExecutionConfigError("[candidate_execution] must be a table")

    backend = _literal(settings, "backend", {"native", "docker"}, "native")
    docker_image = _string(settings, "docker_image", DEFAULT_DOCKER_IMAGE)
    docker_enable_gpu = _bool(settings, "docker_enable_gpu", False)
    docker_user = _optional_string(settings, "docker_user")
    docker_rootless_container_root = _bool(settings, "docker_rootless_container_root", False)
    data_root = _optional_path(settings, "data_root", root)
    max_samples = _optional_int(settings, "max_samples", minimum=1)
    max_prediction_samples = _int(settings, "max_prediction_samples", 2, minimum=0)
    prediction_sample_policy = _literal(
        settings,
        "prediction_sample_policy",
        {"first_n", "adjacent_and_scattered"},
        "first_n",
    )

    if backend == "native":
        if docker_enable_gpu:
            raise CandidateExecutionConfigError("candidate_execution.docker_enable_gpu requires backend = \"docker\"")
        if docker_user is not None:
            raise CandidateExecutionConfigError("candidate_execution.docker_user requires backend = \"docker\"")
        if docker_rootless_container_root:
            raise CandidateExecutionConfigError(
                "candidate_execution.docker_rootless_container_root requires backend = \"docker\""
            )
    if docker_user is not None and docker_rootless_container_root:
        raise CandidateExecutionConfigError(
            "choose either candidate_execution.docker_user or candidate_execution.docker_rootless_container_root, not both"
        )

    return CandidateExecutionConfig(
        backend=backend,  # type: ignore[arg-type]
        docker_image=docker_image,
        docker_enable_gpu=docker_enable_gpu,
        docker_user=docker_user,
        docker_rootless_container_root=docker_rootless_container_root,
        data_root=data_root,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,  # type: ignore[arg-type]
    )


def execution_backend_from_config(config: CandidateExecutionConfig) -> ExecutionBackend:
    """Construct the Harness execution backend selected by config."""

    from ml_autoresearch.execution import DockerBackend, NativeBackend

    if config.backend == "native":
        return NativeBackend()
    return DockerBackend(
        config.docker_image,
        enable_gpu=config.docker_enable_gpu,
        container_user=config.docker_user,
        rootless_container_root=config.docker_rootless_container_root,
    )


def _literal(settings: dict[str, object], key: str, allowed: set[str], default: str) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must be one of: {', '.join(sorted(allowed))}")
    return value


def _string(settings: dict[str, object], key: str, default: str) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value:
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must be a non-empty string")
    return value


def _optional_string(settings: dict[str, object], key: str) -> str | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must be a non-empty string")
    return value


def _bool(settings: dict[str, object], key: str, default: bool) -> bool:
    value = settings.get(key, default)
    if not isinstance(value, bool):
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must be a boolean")
    return value


def _int(settings: dict[str, object], key: str, default: int, *, minimum: int) -> int:
    value = settings.get(key, default)
    if not isinstance(value, int) or value < minimum:
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must be an integer >= {minimum}")
    return value


def _optional_int(settings: dict[str, object], key: str, *, minimum: int) -> int | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value < minimum:
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must be an integer >= {minimum}")
    return value


def _optional_path(settings: dict[str, object], key: str, project_root: Path) -> Path | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path
