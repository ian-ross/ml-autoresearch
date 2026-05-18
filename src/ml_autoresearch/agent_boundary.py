"""Host-side preparation for the Agent Control Boundary."""

from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


REFERENCE_FILES = ("CONTEXT.md", "EXPERIMENT_INDEX.md")
HISTORY_DIRS = ("candidates", "runs", "research-notes")
WORKSPACE_DIRS = (
    "drafts/candidates",
    "submissions",
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
    return AgentBoundaryConfig(
        distro=distro,
        image=image,
        allow_egress=allow_egress,
        data_mounts=_load_data_mounts(data.get("data_mounts", []), project_root),
    )


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
    _refresh_history_snapshot(project_root, history_dir)
    _ensure_workspace(workspace_dir)
    _write_managed_fort_config(project_root, workspace_dir, config)

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


def _refresh_history_snapshot(project_root: Path, history_dir: Path) -> None:
    _clear_snapshot_contents(history_dir)
    ledger = project_root / "research-ledger.jsonl"
    if not ledger.is_file():
        raise AgentBoundaryError(f"missing Research Ledger: {ledger}")
    shutil.copy2(ledger, history_dir / "research-ledger.jsonl")
    for dirname in HISTORY_DIRS:
        (history_dir / dirname).mkdir(parents=True)


def _ensure_workspace(workspace_dir: Path) -> None:
    for dirname in WORKSPACE_DIRS:
        (workspace_dir / dirname).mkdir(parents=True, exist_ok=True)


def _write_managed_fort_config(project_root: Path, workspace_dir: Path, config: AgentBoundaryConfig) -> None:
    pi_dir = workspace_dir / ".pi"
    fort_d = pi_dir / "fort.d"
    pi_dir.mkdir(parents=True, exist_ok=True)
    if fort_d.exists():
        shutil.rmtree(fort_d)
    fort_d.mkdir()
    (fort_d / "README.md").write_text("Managed by `ml-autoresearch prepare-agent-boundary`; contents may be replaced.\n")
    (pi_dir / "fort.toml").write_text(_render_fort_toml(project_root, config))


def _render_fort_toml(project_root: Path, config: AgentBoundaryConfig) -> str:
    mounts = [
        (project_root / "agent-reference", "/reference"),
        (project_root / "agent-history", "/history"),
        (project_root / "agent-history" / "candidates", "/history/candidates"),
        (project_root / "agent-history" / "runs", "/history/runs"),
        (project_root / "agent-history" / "research-notes", "/history/research-notes"),
        (project_root / "docs", "/docs"),
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
