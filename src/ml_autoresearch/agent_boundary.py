"""Host-side preparation for the Agent Control Boundary."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_autoresearch.candidate_execution_config import CandidateExecutionConfig, load_candidate_execution_config
from ml_autoresearch.research_problems import (
    LoadedResearchProblemSpec,
    ResearchProblemProviderLoadError,
    load_research_problem_provider,
)


class AgentBoundaryError(ValueError):
    """Raised when Agent Control Boundary preparation cannot continue."""


@dataclass(frozen=True)
class DataMount:
    """A configured read-only Research Problem data mount."""

    path: Path
    target: str


@dataclass(frozen=True)
class AgentBoundaryConfig:
    """Root agent-boundary.toml settings used to prepare the boundary."""

    distro: str
    image: str
    allow_egress: bool
    data_mounts: tuple[DataMount, ...]
    runs_root: Path
    research_problem: LoadedResearchProblemSpec
    research_problem_provider: Any


REFERENCE_FILES = ("CONTEXT.md", "EXPERIMENT_INDEX.md")
HISTORY_DIRS = ("candidates", "runs", "batches", "research-notes")
WORKSPACE_DIRS = (
    "drafts/candidates",
    "submissions",
    "batch-submissions",
    "research-notes",
    "capability-requests",
    "evaluation-requests",
    "campaign-reports",
    "scratch",
)


def load_agent_boundary_config(project_root: Path) -> AgentBoundaryConfig:
    """Load and validate root agent-boundary.toml."""

    config_path = project_root / "agent-boundary.toml"
    if not config_path.is_file():
        raise AgentBoundaryError(f"missing Agent Control Boundary config: {config_path}")
    data = tomllib.loads(config_path.read_text())
    settings = data.get("agent_control_boundary", {})
    if not isinstance(settings, dict):
        raise AgentBoundaryError("[agent_control_boundary] must be a table")
    distro = _string_setting(settings, "distro", "debian")
    image = _string_setting(settings, "image", "../../containers/ml-autoresearch-agent")
    allow_egress = settings.get("allow_egress", True)
    if not isinstance(allow_egress, bool):
        raise AgentBoundaryError("agent_control_boundary.allow_egress must be a boolean")
    candidate_config = load_candidate_execution_config(project_root)
    return AgentBoundaryConfig(
        distro=distro,
        image=image,
        allow_egress=allow_egress,
        data_mounts=_load_data_mounts(data.get("data_mounts", []), project_root),
        runs_root=candidate_config.runs_root,
        research_problem=_load_agent_boundary_research_problem(candidate_config),
        research_problem_provider=candidate_config.research_problem_provider,
    )


def _load_agent_boundary_research_problem(candidate_config: CandidateExecutionConfig) -> LoadedResearchProblemSpec:
    """Load the explicit Research Problem provider for agent-facing handoff context."""

    provider_config = candidate_config.research_problem_provider
    if provider_config is None:
        raise AgentBoundaryError(
            "Agent Control Boundary handoff/autonomy flows require an explicit [research_problem] "
            "provider in candidate-execution.toml; built-in/default Research Problem fallback is not allowed"
        )
    try:
        return load_research_problem_provider(provider_config)
    except ResearchProblemProviderLoadError as exc:
        raise AgentBoundaryError(f"failed to load configured Research Problem provider: {exc}") from exc


def _string_setting(settings: dict[str, Any], key: str, default: str) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value:
        raise AgentBoundaryError(f"agent_control_boundary.{key} must be a non-empty string")
    return value


def _load_data_mounts(raw_mounts: object, project_root: Path) -> tuple[DataMount, ...]:
    if not isinstance(raw_mounts, list):
        raise AgentBoundaryError("data_mounts must be an array of tables")
    mounts: list[DataMount] = []
    targets: set[str] = set()
    for index, raw_mount in enumerate(raw_mounts, start=1):
        if not isinstance(raw_mount, dict):
            raise AgentBoundaryError("each data mount must be a table")
        name = raw_mount.get("name")
        if not isinstance(name, str) or not name:
            raise AgentBoundaryError(f"data mount {index} name must be a non-empty string")
        path_value = raw_mount.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise AgentBoundaryError(f"data mount {name} path must be a non-empty string")
        readonly = raw_mount.get("readonly", True)
        if readonly is not True:
            raise AgentBoundaryError(f"data mount {name} must be read-only")
        target = raw_mount.get("target", f"/data/{name}")
        if not isinstance(target, str) or not target:
            raise AgentBoundaryError(f"data mount {name} target must be a non-empty string")
        _validate_data_target(target)
        if target in targets:
            raise AgentBoundaryError(f"overlapping data mount target: {target}")
        targets.add(target)
        source_path = Path(path_value).expanduser()
        if not source_path.is_absolute():
            source_path = project_root / source_path
        if not source_path.exists():
            raise AgentBoundaryError(f"data mount path does not exist: {source_path}")
        mounts.append(DataMount(path=source_path, target=target))
    return tuple(mounts)


def _validate_data_target(target: str) -> None:
    parts = Path(target).parts
    if len(parts) != 3 or parts[0] != "/" or parts[1] != "data" or not parts[2]:
        raise AgentBoundaryError("data mount targets must be non-overlapping direct children of /data")


def prepare_agent_boundary(project_root: Path = Path(".")) -> dict[str, str]:
    """Create or refresh generated Agent Control Boundary host-side layout."""

    project_root = project_root.resolve()
    config = load_agent_boundary_config(project_root)
    reference_dir = project_root / "agent-reference"
    history_dir = project_root / "agent-history"
    workspace_dir = project_root / "agent-work"

    _refresh_reference_snapshot(project_root, reference_dir)
    _refresh_history_snapshot(project_root, history_dir, config.runs_root)
    _ensure_workspace(workspace_dir)
    _write_research_problem_brief_index(workspace_dir, config.research_problem)
    _write_agent_workspace_instructions(workspace_dir, config.research_problem)
    _write_agent_candidate_execution_config(workspace_dir, config)
    _write_managed_fort_config(project_root, workspace_dir, config)
    _install_autoresearch_skills(project_root, workspace_dir)
    _install_pi_fort_extension(workspace_dir)

    return {
        "agent_reference": str(reference_dir),
        "agent_history": str(history_dir),
        "agent_workspace": str(workspace_dir),
        "fort_config": str(workspace_dir / ".pi" / "fort.toml"),
    }


def _refresh_reference_snapshot(project_root: Path, reference_dir: Path) -> None:
    _clear_snapshot_contents(reference_dir)
    for filename in REFERENCE_FILES:
        source = project_root / filename
        if not source.is_file():
            raise AgentBoundaryError(f"missing reference file: {source}")
        shutil.copy2(source, reference_dir / filename)


def _refresh_history_snapshot(project_root: Path, history_dir: Path, runs_root: Path) -> None:
    _clear_snapshot_contents(history_dir)
    ledger = project_root / "research-ledger.jsonl"
    if not ledger.is_file():
        raise AgentBoundaryError(f"missing Research Ledger: {ledger}")
    shutil.copy2(ledger, history_dir / "research-ledger.jsonl")
    default_runs_root = project_root / "runs"
    for dirname in HISTORY_DIRS:
        source = runs_root if dirname == "runs" else project_root / dirname
        destination = history_dir / dirname
        if dirname == "runs" and source != default_runs_root:
            source.mkdir(parents=True, exist_ok=True)
            destination.mkdir(parents=True)
        elif source.exists():
            if not source.is_dir():
                raise AgentBoundaryError(f"history source path is not a directory: {source}")
            shutil.copytree(source, destination)
        else:
            destination.mkdir(parents=True)


def _ensure_workspace(workspace_dir: Path) -> None:
    for dirname in WORKSPACE_DIRS:
        (workspace_dir / dirname).mkdir(parents=True, exist_ok=True)


def _write_research_problem_brief_index(workspace_dir: Path, research_problem: LoadedResearchProblemSpec) -> None:
    (workspace_dir / "RESEARCH_PROBLEM_BRIEF_INDEX.md").write_text(
        "# Research Problem Brief index\n\n" + _render_research_problem_brief_index(research_problem) + "\n"
    )


def _render_research_problem_brief_index(research_problem: LoadedResearchProblemSpec) -> str:
    lines = [
        f"Active Research Problem: `{research_problem.spec.id}`",
        f"Spec version: `{research_problem.spec.version}`; contract version: `{research_problem.spec.contract_version}`",
        "Provider package mount: `/research-problem/`",
        "",
        "Use progressive disclosure: start from this index, then read only the deeper brief documents relevant to the current Candidate Experiment.",
    ]
    if not research_problem.brief_documents:
        lines.extend(["", "No Research Problem Brief documents were declared by the configured provider."])
        return "\n".join(lines)
    lines.extend(["", "Available documents:"])
    for document in research_problem.brief_documents:
        mounted_path = Path("/research-problem") / document.path
        summary = document.summary or "No summary provided."
        required = " Required." if document.required else ""
        lines.extend(
            [
                f"- **{document.name}** (`{document.role}`): {summary}{required}",
                f"  - Path: `{mounted_path.as_posix()}`",
                f"  - Read with: `cat {mounted_path.as_posix()}`",
            ]
        )
    return "\n".join(lines)


def _write_agent_candidate_execution_config(workspace_dir: Path, config: AgentBoundaryConfig) -> None:
    provider = config.research_problem_provider
    if provider is None:
        raise AgentBoundaryError("missing configured Research Problem provider for Agent Workspace config")
    data_config = _agent_workspace_data_config(provider.data_config, config.data_mounts)
    lines = [
        "# Generated by ml-autoresearch prepare-agent-boundary; do not edit by hand.",
        "[research_problem]",
        f'id = "{_toml_escape(provider.id)}"',
        'package_root = "/research-problem"',
        f'provider_target = "{_toml_escape(provider.provider_target)}"',
        f'expected_contract_version = "{_toml_escape(provider.expected_contract_version)}"',
    ]
    if data_config:
        entries = ", ".join(f'{key} = "{_toml_escape(str(value))}"' for key, value in sorted(data_config.items()))
        lines.append(f"data_config = {{ {entries} }}")
    (workspace_dir / "candidate-execution.toml").write_text("\n".join(lines) + "\n")


def _agent_workspace_data_config(data_config: dict[str, object], data_mounts: tuple[DataMount, ...]) -> dict[str, object]:
    rewritten: dict[str, object] = {}
    mount_targets = {str(mount.path): mount.target for mount in data_mounts}
    for key, value in data_config.items():
        rewritten[key] = mount_targets.get(str(value), value) if isinstance(value, str) else value
    return rewritten


def _write_agent_workspace_instructions(workspace_dir: Path, research_problem: LoadedResearchProblemSpec) -> None:
    brief_index = _render_research_problem_brief_index(research_problem)
    (workspace_dir / "AGENTS.md").write_text(
        "# Agent Control Boundary path map\n"
        "\n"
        "You are running inside the Agent Workspace. Autoresearch skills may refer to\n"
        "project-root paths from the outer Harness repository; translate those paths to\n"
        "the mounted paths inside this boundary.\n"
        "\n"
        "## Read-only reference and history\n"
        "\n"
        "- `CONTEXT.md` -> `/reference/CONTEXT.md`\n"
        "- `EXPERIMENT_INDEX.md` -> `/reference/EXPERIMENT_INDEX.md`\n"
        "- `docs/` -> `/docs/`\n"
        "- `research-ledger.jsonl` -> `/history/research-ledger.jsonl`\n"
        "- `candidates/` -> `/history/candidates/` for prior Candidate sources\n"
        "- `runs/` -> `/history/runs/` for prior Run summaries/artifacts\n"
        "- `batches/` -> `/history/batches/` for prior Experiment Batch summaries/artifacts\n"
        "- `research-notes/` -> `/history/research-notes/` for prior notes\n"
        "- `/data/` contains approved read-only Research Problem data mounts when present\n"
        "\n"
        "## Active Research Problem Brief\n"
        "\n"
        + brief_index
        + "\n\n"
        "The same index is available at `RESEARCH_PROBLEM_BRIEF_INDEX.md`. Read only the deeper `/research-problem/...` documents you need.\n"
        "\n"
        "## Dataset profile artifacts\n"
        "\n"
        "Dataset profile artifacts, when supplied by the Harness or active Research Problem package, are trusted agent-visible context under `/research-problem/profile/` or linked from `RESEARCH_PROBLEM_BRIEF_INDEX.md`. Use them as reproducible dataset intelligence for proposals and analysis; they are not raw training data or authoritative Run Results. If a needed statistic or qualitative view is missing, write a Capability Request instead of probing `/data`.\n"
        "\n"
        "## Writable handoff locations\n"
        "\n"
        "- Draft Candidate Experiments: `drafts/candidates/`\n"
        "- Final Candidate Submission Queue entries: `submissions/`\n"
        "- Final Experiment Batch Submission Queue entries: `batch-submissions/`\n"
        "- write new draft Research Notes under `research-notes/`\n"
        "- Capability Requests: `capability-requests/`\n"
        "- Evaluation Requests: `evaluation-requests/`\n"
        "- Campaign Reports: `campaign-reports/`\n"
        "- Scratch files: `scratch/`\n"
        "\n"
        "One Autonomy Step means one primary handoff outcome, then stop. Do not\n"
        "produce a second Candidate Submission, Experiment Batch Submission, Research Note,\n"
        "Capability Request, Evaluation Request, or Campaign Report in the same step.\n"
        "\n"
        "Use `ml-autoresearch-agent`, not `ml-autoresearch`, for allowed observation\n"
        "and static Candidate preparation commands. Observation commands default to\n"
        "the `/history/runs` and `/history/batches` Research History roots; do not\n"
        "invent host-relative Runs root paths. Do not edit mounted read-only\n"
        "reference, history, docs, or data paths.\n"
    )


def _write_managed_fort_config(project_root: Path, workspace_dir: Path, config: AgentBoundaryConfig) -> None:
    pi_dir = workspace_dir / ".pi"
    fort_d = pi_dir / "fort.d"
    pi_dir.mkdir(parents=True, exist_ok=True)
    if fort_d.exists():
        shutil.rmtree(fort_d)
    fort_d.mkdir()
    (fort_d / "README.md").write_text("Managed by `ml-autoresearch prepare-agent-boundary`; contents may be replaced.\n")
    (pi_dir / "fort.toml").write_text(_render_fort_toml(project_root, config))


def _install_autoresearch_skills(project_root: Path, workspace_dir: Path) -> None:
    source_dir = project_root / "docs" / "autoresearch-skills"
    if not source_dir.exists():
        return
    if not source_dir.is_dir():
        raise AgentBoundaryError(f"Autoresearch Skill Set path is not a directory: {source_dir}")
    skills_dir = workspace_dir / ".pi" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for source_entry in source_dir.iterdir():
        destination = skills_dir / source_entry.name
        if destination.exists():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if source_entry.is_dir():
            shutil.copytree(source_entry, destination)
        else:
            shutil.copy2(source_entry, destination)


def _install_pi_fort_extension(workspace_dir: Path) -> None:
    pi_fort_path = _resolve_pi_fort_path_from_environment()
    command = ["pi", "install", "-l", str(pi_fort_path)]
    try:
        completed = subprocess.run(command, cwd=workspace_dir, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise AgentBoundaryError(
            "failed to install pi-fort into Agent Workspace; `pi` executable was not found\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {workspace_dir}"
        ) from exc
    except OSError as exc:
        raise AgentBoundaryError(
            "failed to install pi-fort into Agent Workspace\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {workspace_dir}\n"
            f"error: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise AgentBoundaryError(
            "failed to install pi-fort into Agent Workspace\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {workspace_dir}\n"
            f"exit code: {completed.returncode}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )


def _resolve_pi_fort_path_from_environment() -> Path:
    raw_path = os.environ.get("ML_AUTORESEARCH_PI_FORT")
    if raw_path is None or not raw_path.strip():
        raise AgentBoundaryError("ML_AUTORESEARCH_PI_FORT must be set to an absolute local path to pi-fort")
    expanded = Path(raw_path).expanduser()
    if not expanded.is_absolute():
        raise AgentBoundaryError("ML_AUTORESEARCH_PI_FORT must be an absolute local path after ~ expansion")
    try:
        return expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AgentBoundaryError(f"ML_AUTORESEARCH_PI_FORT path does not exist: {expanded}") from exc
    except OSError as exc:
        raise AgentBoundaryError(f"ML_AUTORESEARCH_PI_FORT path cannot be resolved: {expanded}: {exc}") from exc


def _render_fort_toml(project_root: Path, config: AgentBoundaryConfig) -> str:
    mounts = [
        (project_root / "agent-reference", "/reference"),
        (project_root / "agent-history", "/history"),
        (project_root / "agent-history" / "candidates", "/history/candidates"),
        (_runs_history_mount_source(project_root, config.runs_root), "/history/runs"),
        (project_root / "agent-history" / "batches", "/history/batches"),
        (project_root / "agent-history" / "research-notes", "/history/research-notes"),
        (project_root / "docs", "/docs"),
        (config.research_problem.provenance.resolved_package_root, "/research-problem"),
        (project_root / "src" / "ml_autoresearch", "/usr/local/lib/python3.12/site-packages/ml_autoresearch"),
        (project_root / "src" / "ml_autoresearch", "/usr/local/lib/python3.12/dist-packages/ml_autoresearch"),
    ]
    mount_entries = [_format_mount(path, target) for path, target in mounts]
    mount_entries.extend(_format_mount(mount.path, mount.target) for mount in config.data_mounts)
    return (
        "# Generated by ml-autoresearch prepare-agent-boundary; do not edit by hand.\n"
        "enabled = true\n"
        f"allow_egress = {_toml_bool(config.allow_egress)}\n"
        f"distro = \"{_toml_escape(config.distro)}\"\n"
        f"image = \"{_toml_escape(config.image)}\"\n"
        "mounts = [\n"
        + "\n".join(f"  {entry}," for entry in mount_entries)
        + "\n]\n"
    )


def _runs_history_mount_source(project_root: Path, runs_root: Path) -> Path:
    default_runs_root = project_root / "runs"
    if runs_root != default_runs_root:
        return runs_root
    return project_root / "agent-history" / "runs"


def _format_mount(path: Path, target: str) -> str:
    return f'{{path="{_toml_escape(str(path))}", target="{_toml_escape(target)}", readonly=true}}'


def _clear_snapshot_contents(snapshot_dir: Path) -> None:
    if snapshot_dir.exists():
        if not snapshot_dir.is_dir():
            raise AgentBoundaryError(f"snapshot path is not a directory: {snapshot_dir}")
        for entry in snapshot_dir.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    else:
        snapshot_dir.mkdir(parents=True)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_escape(value: str) -> str:
    return json.dumps(value)[1:-1]
