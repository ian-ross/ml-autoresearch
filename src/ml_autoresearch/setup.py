"""Research Workspace Setup implementation."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ml_autoresearch.workspace import CANONICAL_RESEARCH_LEDGER, WORKSPACE_CONFIG_FILENAME


class WorkspaceSetupError(ValueError):
    """Raised when Research Workspace Setup cannot proceed safely."""


SUPPORTED_BINARY_SEGMENTATION = "binary_semantic_segmentation"


@dataclass(frozen=True)
class WorkspaceSetupRequest:
    """Inputs for initializing a Research Workspace Root."""

    workspace_root: Path = Path(".")
    problem_id: str = "research_problem"
    provider_module: str = "research_problem"
    problem_type: str = SUPPORTED_BINARY_SEGMENTATION
    runs_root: str = "runs"
    reset_research_memory: bool = False


@dataclass(frozen=True)
class WorkspaceSetupResult:
    """Summary of created/skipped Research Workspace paths."""

    workspace_root: Path
    created: tuple[Path, ...]
    skipped: tuple[Path, ...]
    runs_root: Path
    provider_target: str


def infer_provider_module_from_pyproject(workspace_root: str | Path) -> str:
    """Infer a Python import package name from ``pyproject.toml`` project metadata."""

    root = Path(workspace_root)
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return "research_problem"
    try:
        data = tomllib.loads(pyproject.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise WorkspaceSetupError(f"invalid pyproject.toml: {exc}") from exc
    project = data.get("project", {})
    if not isinstance(project, dict):
        return "research_problem"
    name = project.get("name")
    if not isinstance(name, str) or not name:
        return "research_problem"
    return _module_name(name)


def initialize_workspace(request: WorkspaceSetupRequest) -> WorkspaceSetupResult:
    """Create missing Research Workspace files and directories conservatively."""

    root = request.workspace_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _validate_import_module(request.provider_module)
    _validate_problem_id(request.problem_id)
    runs_root = _resolve_under_workspace(root, request.runs_root)
    provider_file = root.joinpath(*request.provider_module.split("."), "research_problem.py")
    provider_target = f"{request.provider_module}.research_problem:build_spec"

    created: list[Path] = []
    skipped: list[Path] = []

    def write_missing(relative: str, content: str, *, allow_existing: bool = True) -> None:
        path = root / relative
        if path.exists():
            if not path.is_file():
                raise WorkspaceSetupError(f"expected file but found directory: {path}")
            if not allow_existing and path.read_text() != content:
                raise WorkspaceSetupError(f"refusing to overwrite existing file: {path}")
            skipped.append(path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        created.append(path)

    def mkdir_gitkeep(relative: str) -> None:
        directory = root / relative
        if not directory.exists():
            directory.mkdir(parents=True)
            created.append(directory)
        elif not directory.is_dir():
            raise WorkspaceSetupError(f"expected directory but found file: {directory}")
        gitkeep = directory / ".gitkeep"
        if gitkeep.exists():
            skipped.append(gitkeep)
        else:
            gitkeep.write_text("")
            created.append(gitkeep)

    write_missing(
        WORKSPACE_CONFIG_FILENAME,
        _workspace_config(request, provider_target=provider_target),
        allow_existing=False,
    )
    write_missing("CONTEXT.md", _context_markdown(request))
    write_missing("AGENTS.md", _agents_markdown())
    write_missing("EXPERIMENT_INDEX.md", _experiment_index())
    if (root / CANONICAL_RESEARCH_LEDGER).exists() and request.reset_research_memory:
        (root / CANONICAL_RESEARCH_LEDGER).write_text("")
        created.append(root / CANONICAL_RESEARCH_LEDGER)
    else:
        write_missing(CANONICAL_RESEARCH_LEDGER, "")
    write_missing("brief/overview.md", _brief_overview(request))
    write_missing("brief/baselines.md", _brief_baselines())
    write_missing("profile/dataset-profile.md", _dataset_profile())
    write_missing(".gitignore", _gitignore(request.runs_root))

    package_path = root
    for package_part in request.provider_module.split("."):
        package_path = package_path / package_part
        write_missing(str(package_path.relative_to(root) / "__init__.py"), "")
    write_missing(
        str(provider_file.relative_to(root)),
        _provider_code(request),
        allow_existing=False,
    )

    for directory in _canonical_directories():
        mkdir_gitkeep(directory)
    runs_root.mkdir(parents=True, exist_ok=True)
    if not runs_root.is_dir():
        raise WorkspaceSetupError(f"runs_root is not a directory: {runs_root}")
    created.append(runs_root)

    return WorkspaceSetupResult(
        workspace_root=root,
        created=tuple(created),
        skipped=tuple(skipped),
        runs_root=runs_root,
        provider_target=provider_target,
    )


def _module_name(project_name: str) -> str:
    module = re.sub(r"[^0-9A-Za-z_]+", "_", project_name).strip("_").lower()
    if not module:
        return "research_problem"
    if module[0].isdigit():
        module = f"_{module}"
    return module


def _validate_import_module(module: str) -> None:
    parts = module.split(".")
    if any(not re.fullmatch(r"[A-Za-z_]\w*", part) for part in parts):
        raise WorkspaceSetupError(f"provider module is not a valid Python import path: {module}")


def _validate_problem_id(problem_id: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", problem_id):
        raise WorkspaceSetupError("problem id must use lowercase letters, numbers, and underscores, starting with a letter")


def _resolve_under_workspace(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _workspace_config(request: WorkspaceSetupRequest, *, provider_target: str) -> str:
    return f'''# Workspace Configuration generated by `ml-autoresearch setup`.

[research_problem]
id = "{request.problem_id}"
package_root = "."
provider_target = "{provider_target}"
expected_contract_version = "v0"
data_config = {{ dataset_root = "data" }}

[candidate_execution]
backend = "native"
runs_root = "{request.runs_root}"
ledger_path = "research-ledger.jsonl"
max_prediction_samples = 2
prediction_sample_policy = "first_n"
'''


def _context_markdown(request: WorkspaceSetupRequest) -> str:
    return f"""# {request.problem_id}

This Research Problem Repository was initialized by `ml-autoresearch setup`.

Fill in the domain language, data source, prediction target, evaluation metrics,
and constraints before running real Candidate Experiments.
"""


def _agents_markdown() -> str:
    return """# Agent instructions

This is a Research Problem Repository and Research Workspace Root.

Repository maintenance agents may edit trusted problem code and documentation.
Generated Agent Workspace instructions under `agent-work/` are narrower: agents
inside the Agent Control Boundary must use `ml-autoresearch-agent` and must not
change Harness or trusted Research Problem policy directly.
"""


def _experiment_index() -> str:
    return """# Experiment Index

No Candidate Experiments or Research Notes have been recorded yet.
"""


def _brief_overview(request: WorkspaceSetupRequest) -> str:
    return f"""# Research Problem Brief: Overview

Problem id: `{request.problem_id}`

TODO: Describe the data modality, prediction target, operational constraints,
and what useful Research Progress means for this Research Problem.
"""


def _brief_baselines() -> str:
    return """# Research Problem Brief: Baselines

TODO: Record known baselines, prior work, and evaluation caveats.
"""


def _dataset_profile() -> str:
    return """# Dataset Profile Artifact

TODO: Generate or record trusted dataset-profile summaries such as class balance,
mask-area distribution, split caveats, and qualitative example selection policy.
"""


def _gitignore(runs_root: str) -> str:
    root = runs_root.rstrip("/") + "/"
    return f"""# Python
__pycache__/
*.py[cod]
.venv/
.env

# ML Autoresearch local operational state
.ml-autoresearch/
{root}
"""


def _provider_code(request: WorkspaceSetupRequest) -> str:
    if request.problem_type != SUPPORTED_BINARY_SEGMENTATION:
        return f'''"""Starter Research Problem Spec provider."""


def build_spec(data_config=None):
    """Return a ResearchProblemSpec once this problem type is implemented."""

    raise NotImplementedError(
        "Unsupported Research Problem type {request.problem_type!r}. TODO: implement a trusted "
        "ResearchProblemSpec provider for this Research Problem before running Candidate Experiments."
    )
'''
    return f'''"""Starter Research Problem Spec provider for binary semantic segmentation."""

from ml_autoresearch.research_problems import (
    DatasetProfileArtifact,
    ResearchProblemBriefDocument,
    ResearchProblemSpec,
)


def build_spec(data_config=None):
    """Build the trusted Research Problem Spec for this workspace.

    The defaults are intentionally conservative binary semantic segmentation
    placeholders. Replace dimensions, adapters, and allowed policies with
    problem-specific trusted infrastructure before real Runs.
    """

    return ResearchProblemSpec(
        id={request.problem_id!r},
        version="starter-v0",
        contract_version="v0",
        input_modes=("single_frame_rgb",),
        input_specs={{"single_frame_rgb": {{"mode": "single_frame_rgb", "shape": [3, 256, 256]}}}},
        output_forms=("mask_logits",),
        output_specs={{"mask_logits": {{"form": "mask_logits", "shape": [1, 256, 256]}}}},
        losses=("bce_dice",),
        optimizers=("adamw",),
        sampling_policies=("sequential",),
        augmentation_policies=("none",),
        primary_metric="val/dice",
        brief_documents=(
            ResearchProblemBriefDocument(
                name="overview",
                role="problem_overview",
                path="brief/overview.md",
                summary="Starter Research Problem overview.",
                required=True,
            ),
            ResearchProblemBriefDocument(
                name="baselines",
                role="baseline_description",
                path="brief/baselines.md",
                summary="Starter baseline and literature notes.",
            ),
        ),
        dataset_profile_artifacts=(
            DatasetProfileArtifact(
                name="starter_dataset_profile",
                role="trusted_dataset_profile",
                path="profile/dataset-profile.md",
                summary="Starter dataset profile placeholder.",
            ),
        ),
    )
'''


def _canonical_directories() -> tuple[str, ...]:
    return (
        ".ml-autoresearch",
        "candidates",
        "experiment-batches",
        "research-notes",
        "capability-requests",
        "evaluation-requests",
        "campaign-reports",
        "agent-work/submissions",
        "agent-work/batch-submissions",
        "agent-work/research-notes",
        "agent-work/capability-requests",
        "agent-work/evaluation-requests",
        "agent-work/campaign-reports",
    )
