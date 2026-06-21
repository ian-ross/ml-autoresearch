"""Research Workspace Root configuration and fixed workspace paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WORKSPACE_CONFIG_FILENAME = "ml-autoresearch.toml"
CANONICAL_RESEARCH_LEDGER = "research-ledger.jsonl"


class WorkspaceConfigError(ValueError):
    """Raised when Workspace Configuration cannot be loaded."""


@dataclass(frozen=True)
class WorkspacePaths:
    """Centralized Research Workspace Root-relative durable state paths."""

    workspace_root: Path
    runs_root: Path

    @classmethod
    def from_root(cls, workspace_root: str | Path, *, runs_root: str | Path | None = None) -> "WorkspacePaths":
        root = Path(workspace_root).resolve()
        effective_runs = _resolve_workspace_path(root, runs_root) if runs_root is not None else root / "runs"
        return cls(workspace_root=root, runs_root=effective_runs)

    @property
    def config_path(self) -> Path:
        return self.workspace_root / WORKSPACE_CONFIG_FILENAME

    @property
    def ledger_path(self) -> Path:
        return self.workspace_root / CANONICAL_RESEARCH_LEDGER

    @property
    def agent_work_root(self) -> Path:
        return self.workspace_root / "agent-work"

    @property
    def candidates_root(self) -> Path:
        return self.workspace_root / "candidates"

    @property
    def batches_root(self) -> Path:
        return self.workspace_root / "batches"

    @property
    def experiment_batches_root(self) -> Path:
        return self.workspace_root / "experiment-batches"

    @property
    def research_notes_root(self) -> Path:
        return self.workspace_root / "research-notes"

    @property
    def capability_requests_root(self) -> Path:
        return self.workspace_root / "capability-requests"

    @property
    def evaluation_requests_root(self) -> Path:
        return self.workspace_root / "evaluation-requests"

    @property
    def campaign_reports_root(self) -> Path:
        return self.workspace_root / "campaign-reports"

    @property
    def experiment_index_path(self) -> Path:
        return self.workspace_root / "EXPERIMENT_INDEX.md"


def resolve_workspace_root(workspace_root: str | Path = Path(".")) -> Path:
    """Resolve the Research Workspace Root supplied by a CLI or API caller."""

    return Path(workspace_root).resolve()


def workspace_config_path(workspace_root: str | Path = Path(".")) -> Path:
    """Return the canonical Workspace Configuration path under a Research Workspace Root."""

    return resolve_workspace_root(workspace_root) / WORKSPACE_CONFIG_FILENAME


def _resolve_workspace_path(workspace_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path
