"""Research Workspace Configuration for Candidate Execution Boundary policy."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, ExecutionBackend
from ml_autoresearch.parameter_budget import DEFAULT_MAX_PARAMETER_COUNT, MAX_CONFIGURABLE_PARAMETER_COUNT
from ml_autoresearch.workspace import WORKSPACE_CONFIG_FILENAME, WorkspaceConfigError
from ml_autoresearch.research_problems import (
    ResearchProblemProviderConfig,
    ResearchProblemSpecRegistry,
    load_research_problem_provider,
)

CONFIG_FILENAME = WORKSPACE_CONFIG_FILENAME


class CandidateExecutionConfigError(WorkspaceConfigError):
    """Raised when Workspace Configuration execution policy is invalid."""


@dataclass(frozen=True)
class CandidateExecutionConfig:
    """Harness-owned execution policy for Candidate Experiment Runs."""

    backend: Literal["native", "docker"] = "native"
    docker_image: str = DEFAULT_DOCKER_IMAGE
    docker_enable_gpu: bool = False
    docker_gpu_device: str | None = None
    docker_user: str | None = None
    docker_rootless_container_root: bool = False
    data_root: Path | None = None
    runs_root: Path = Path("runs")
    ledger_path: Path | None = None
    max_samples: int | None = None
    max_parameters: int = DEFAULT_MAX_PARAMETER_COUNT
    max_prediction_samples: int = 2
    max_parallel_runs: int = 1
    prediction_sample_policy: Literal["first_n", "adjacent_and_scattered"] = "first_n"
    research_problem_provider: ResearchProblemProviderConfig | None = None


def load_candidate_execution_config(workspace_root: str | Path = Path(".")) -> CandidateExecutionConfig:
    """Load Candidate Execution Boundary policy from canonical Workspace Configuration."""

    root = Path(workspace_root).resolve()
    path = root / CONFIG_FILENAME
    if not path.is_file():
        raise CandidateExecutionConfigError(f"missing Workspace Configuration: {path}")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise CandidateExecutionConfigError(f"invalid Workspace Configuration {path}: {exc}") from exc
    settings = data.get("candidate_execution", {})
    if not isinstance(settings, dict):
        raise CandidateExecutionConfigError("[candidate_execution] must be a table")

    backend = _literal(settings, "backend", {"native", "docker"}, "native")
    docker_image = _string(settings, "docker_image", DEFAULT_DOCKER_IMAGE)
    docker_enable_gpu = _bool(settings, "docker_enable_gpu", False)
    docker_gpu_device = _optional_string(settings, "docker_gpu_device")
    if docker_gpu_device is not None and re.fullmatch(
        r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9-]+)", docker_gpu_device
    ) is None:
        raise CandidateExecutionConfigError(
            "candidate_execution.docker_gpu_device must identify one GPU by numeric index, GPU UUID, or MIG UUID"
        )
    docker_user = _optional_string(settings, "docker_user")
    docker_rootless_container_root = _bool(settings, "docker_rootless_container_root", False)
    data_root = _optional_path(settings, "data_root", root)
    if "runs_root" in settings:
        runs_root = _path(settings, "runs_root", root, prefix="candidate_execution")
    else:
        runs_root = root / "runs"
    ledger_path = _optional_workspace_path(settings, "ledger_path", root)
    research_problem_provider = _research_problem_provider_config(data, root)
    if data_root is not None and research_problem_provider is not None and research_problem_provider.data_roots:
        raise CandidateExecutionConfigError(
            "candidate_execution.data_root cannot be combined with research_problem.data_roots"
        )
    max_samples = _optional_int(settings, "max_samples", minimum=1)
    max_parameters = _int(settings, "max_parameters", DEFAULT_MAX_PARAMETER_COUNT, minimum=1)
    if max_parameters > MAX_CONFIGURABLE_PARAMETER_COUNT:
        raise CandidateExecutionConfigError(
            f"candidate_execution.max_parameters must be at most {MAX_CONFIGURABLE_PARAMETER_COUNT}"
        )
    max_prediction_samples = _int(settings, "max_prediction_samples", 2, minimum=0)
    max_parallel_runs = _int(settings, "max_parallel_runs", 1, minimum=1)
    if max_parallel_runs > 4:
        raise CandidateExecutionConfigError("candidate_execution.max_parallel_runs must be at most 4")
    prediction_sample_policy = _literal(
        settings,
        "prediction_sample_policy",
        {"first_n", "adjacent_and_scattered"},
        "first_n",
    )

    if backend == "native":
        if docker_enable_gpu:
            raise CandidateExecutionConfigError("candidate_execution.docker_enable_gpu requires backend = \"docker\"")
        if docker_gpu_device is not None:
            raise CandidateExecutionConfigError("candidate_execution.docker_gpu_device requires backend = \"docker\"")
        if docker_user is not None:
            raise CandidateExecutionConfigError("candidate_execution.docker_user requires backend = \"docker\"")
        if docker_rootless_container_root:
            raise CandidateExecutionConfigError(
                "candidate_execution.docker_rootless_container_root requires backend = \"docker\""
            )
    if docker_gpu_device is not None and not docker_enable_gpu:
        raise CandidateExecutionConfigError("candidate_execution.docker_gpu_device requires docker_enable_gpu = true")
    if docker_user is not None and docker_rootless_container_root:
        raise CandidateExecutionConfigError(
            "choose either candidate_execution.docker_user or candidate_execution.docker_rootless_container_root, not both"
        )

    return CandidateExecutionConfig(
        backend=backend,  # type: ignore[arg-type]
        docker_image=docker_image,
        docker_enable_gpu=docker_enable_gpu,
        docker_gpu_device=docker_gpu_device,
        docker_user=docker_user,
        docker_rootless_container_root=docker_rootless_container_root,
        data_root=data_root,
        runs_root=runs_root,
        ledger_path=ledger_path,
        max_samples=max_samples,
        max_parameters=max_parameters,
        max_prediction_samples=max_prediction_samples,
        max_parallel_runs=max_parallel_runs,
        prediction_sample_policy=prediction_sample_policy,  # type: ignore[arg-type]
        research_problem_provider=research_problem_provider,
    )


def load_configured_research_problem_registry(workspace_root: str | Path = Path(".")) -> ResearchProblemSpecRegistry | None:
    """Load the configured trusted Research Problem Spec Registry, when configured.

    ``None`` preserves compatibility/bootstrap behavior where callers supply
    their own Research Problem registry elsewhere.
    """

    config = load_candidate_execution_config(workspace_root)
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
        gpu_device=config.docker_gpu_device,
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
    if provider.data_roots:
        data_roots = dict(provider.data_roots)
        if data_root is not None:
            if "training" not in data_roots:
                raise CandidateExecutionConfigError(
                    "a data-root override requires research_problem.data_roots.training"
                )
            data_roots["training"] = Path(data_root)
        return provider.model_copy(update={"data_roots": data_roots})
    if data_root is not None:
        if "dataset_root" in data_config:
            data_config["dataset_root"] = str(data_root)
        elif "data_root" in data_config:
            data_config["data_root"] = str(data_root)
        else:
            data_config["dataset_root"] = str(data_root)
    return provider.model_copy(update={"data_config": data_config})


def _research_problem_provider_config(data: dict[str, object], workspace_root: Path) -> ResearchProblemProviderConfig | None:
    settings = data.get("research_problem")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise CandidateExecutionConfigError("[research_problem] must be a table")
    spec_id = _required_string(settings, "id")
    package_root = _path(settings, "package_root", workspace_root)
    provider_target = _required_string(settings, "provider_target")
    expected_contract_version = _string(settings, "expected_contract_version", "v0")
    data_config = settings.get("data_config", {})
    if not isinstance(data_config, dict):
        raise CandidateExecutionConfigError("research_problem.data_config must be a table")
    if "data_roots" in data_config:
        raise CandidateExecutionConfigError(
            "research_problem.data_config.data_roots is reserved; configure [research_problem.data_roots]"
        )
    data_roots = _research_problem_data_roots(settings.get("data_roots", {}), workspace_root)
    return ResearchProblemProviderConfig(
        id=spec_id,
        package_root=package_root,
        provider_target=provider_target,
        expected_contract_version=expected_contract_version,
        data_config=dict(data_config),
        data_roots=data_roots,
    )


def _research_problem_data_roots(raw_roots: object, workspace_root: Path) -> dict[str, Path]:
    if not isinstance(raw_roots, dict):
        raise CandidateExecutionConfigError("research_problem.data_roots must be a table")
    roots: dict[str, Path] = {}
    resolved_sources: dict[Path, str] = {}
    for name, raw_path in sorted(raw_roots.items()):
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", name) is None:
            raise CandidateExecutionConfigError(
                "research_problem.data_roots names must match [a-z][a-z0-9_-]*"
            )
        if not isinstance(raw_path, str) or not raw_path:
            raise CandidateExecutionConfigError(f"research_problem.data_roots.{name} must be a non-empty string")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = workspace_root / path
        if not path.exists():
            raise CandidateExecutionConfigError(f"Research Problem data root {name!r} does not exist: {path}")
        if not path.is_dir():
            raise CandidateExecutionConfigError(f"Research Problem data root {name!r} is not a directory: {path}")
        resolved = path.resolve(strict=True)
        previous_name = resolved_sources.get(resolved)
        if previous_name is not None:
            raise CandidateExecutionConfigError(
                f"Research Problem data roots {previous_name!r} and {name!r} resolve to the same directory: {resolved}"
            )
        resolved_sources[resolved] = name
        roots[name] = resolved
    return roots


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


def _optional_path(settings: dict[str, object], key: str, workspace_root: Path) -> Path | None:
    value = settings.get(key)
    if value is None:
        return None
    return _path(settings, key, workspace_root, prefix="candidate_execution")


def _optional_workspace_path(settings: dict[str, object], key: str, workspace_root: Path) -> Path | None:
    path = _optional_path(settings, key, workspace_root)
    if path is None:
        return None
    resolved_root = workspace_root.resolve()
    resolved_path = path.resolve(strict=False)
    if resolved_path != resolved_root and not resolved_path.is_relative_to(resolved_root):
        raise CandidateExecutionConfigError(f"candidate_execution.{key} must resolve inside the Research Workspace Root")
    return path


def _path(settings: dict[str, object], key: str, workspace_root: Path, *, prefix: str = "research_problem") -> Path:
    value = settings.get(key)
    if not isinstance(value, str) or not value:
        raise CandidateExecutionConfigError(f"{prefix}.{key} must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path
