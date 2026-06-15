"""Project-local Candidate Execution Boundary configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, ExecutionBackend
from ml_autoresearch.research_problems import (
    ResearchProblemProviderConfig,
    ResearchProblemSpecRegistry,
    load_research_problem_provider,
)

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
    runs_root: Path = Path("runs")
    ledger_path: Path | None = None
    max_samples: int | None = None
    max_prediction_samples: int = 2
    prediction_sample_policy: Literal["first_n", "adjacent_and_scattered"] = "first_n"
    research_problem_provider: ResearchProblemProviderConfig | None = None


def load_candidate_execution_config(project_root: str | Path = Path(".")) -> CandidateExecutionConfig:
    """Load Candidate Execution Boundary config from candidate-execution.toml if present."""

    root = Path(project_root).resolve()
    path = root / CONFIG_FILENAME
    if not path.is_file():
        return CandidateExecutionConfig(runs_root=root / "runs")
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
    if "runs_root" in settings:
        runs_root = _path(settings, "runs_root", root, prefix="candidate_execution")
    else:
        runs_root = root / "runs"
    ledger_path = _optional_path(settings, "ledger_path", root)
    research_problem_provider = _research_problem_provider_config(data, root)
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
        runs_root=runs_root,
        ledger_path=ledger_path,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,  # type: ignore[arg-type]
        research_problem_provider=research_problem_provider,
    )


def load_configured_research_problem_registry(project_root: str | Path = Path(".")) -> ResearchProblemSpecRegistry | None:
    """Load the configured trusted Research Problem Spec Registry, when configured.

    ``None`` preserves compatibility/bootstrap behavior where callers supply
    their own Research Problem registry elsewhere.
    """

    config = load_candidate_execution_config(project_root)
    if config.research_problem_provider is None:
        return None
    registry = ResearchProblemSpecRegistry(active_id=config.research_problem_provider.id)
    load_research_problem_provider(config.research_problem_provider, registry=registry)
    return registry


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


def resolve_configured_research_problem_provider(
    config: CandidateExecutionConfig,
    *,
    data_root_override: str | Path | None = None,
) -> ResearchProblemProviderConfig | None:
    """Return the configured Research Problem provider with data-root compatibility applied."""

    provider = config.research_problem_provider
    if provider is None:
        return None
    data_config = dict(provider.data_config)
    data_root = data_root_override if data_root_override is not None else config.data_root
    if data_root is not None:
        if "dataset_root" in data_config:
            data_config["dataset_root"] = str(data_root)
        elif "data_root" in data_config:
            data_config["data_root"] = str(data_root)
        else:
            data_config["dataset_root"] = str(data_root)
    return provider.model_copy(update={"data_config": data_config})


def _research_problem_provider_config(data: dict[str, object], project_root: Path) -> ResearchProblemProviderConfig | None:
    settings = data.get("research_problem")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise CandidateExecutionConfigError("[research_problem] must be a table")
    spec_id = _required_string(settings, "id")
    package_root = _path(settings, "package_root", project_root)
    provider_target = _required_string(settings, "provider_target")
    expected_contract_version = _string(settings, "expected_contract_version", "v0")
    data_config = settings.get("data_config", {})
    if not isinstance(data_config, dict):
        raise CandidateExecutionConfigError("research_problem.data_config must be a table")
    return ResearchProblemProviderConfig(
        id=spec_id,
        package_root=package_root,
        provider_target=provider_target,
        expected_contract_version=expected_contract_version,
        data_config=dict(data_config),
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


def _required_string(settings: dict[str, object], key: str, *, prefix: str = "research_problem") -> str:
    value = settings.get(key)
    if not isinstance(value, str) or not value:
        raise CandidateExecutionConfigError(f"{prefix}.{key} must be a non-empty string")
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
    return _path(settings, key, project_root, prefix="candidate_execution")


def _path(settings: dict[str, object], key: str, project_root: Path, *, prefix: str = "research_problem") -> Path:
    value = settings.get(key)
    if not isinstance(value, str) or not value:
        raise CandidateExecutionConfigError(f"{prefix}.{key} must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path
