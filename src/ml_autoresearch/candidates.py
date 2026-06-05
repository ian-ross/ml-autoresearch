"""Candidate Experiment contract validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml_autoresearch.research_problems import (
    DEFAULT_RESEARCH_PROBLEM_ID,
    ResearchProblemSpec,
    ResearchProblemSpecRegistry,
    UnknownResearchProblemSpecError,
    get_default_research_problem_spec,
)


class CandidateValidationError(ValueError):
    """Raised when a Candidate Experiment does not satisfy the v1 contract."""


class TrainingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: str = Field(min_length=1)
    optimizer: str = Field(min_length=1)
    learning_rate: float = Field(ge=1e-5, le=3e-3)
    batch_size: int = Field(ge=1, le=32)
    max_epochs: int = Field(ge=1, le=100)


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sampling_policy: str = Field(default="sequential", min_length=1)
    augmentation_policy: str = Field(default="none", min_length=1)


class AuxiliaryTargetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    output: str = Field(min_length=1)
    loss: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)


class RepairLineage(BaseModel):
    """Structured lineage for a Repair Candidate.

    Repair Candidates may fix implementation defects while preserving the
    original Experiment Proposal hypothesis and Comparison Target.
    """

    model_config = ConfigDict(extra="forbid")

    original_proposal_id: str = Field(min_length=1)
    original_candidate_id: str = Field(min_length=1)
    motivating_run_id: str = Field(min_length=1)
    failure_classification: Literal[
        "candidate_bug",
        "contract_violation",
        "resource_failure",
        "harness_failure",
        "bad_research_result",
        "unknown",
    ]
    preserves_original_hypothesis: Literal[True]
    preserves_comparison_target: Literal[True]



class CandidateManifest(BaseModel):
    """Normalized v1 Candidate Experiment source manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    research_problem: str = Field(default=DEFAULT_RESEARCH_PROBLEM_ID, min_length=1)
    input_mode: str = Field(min_length=1)
    output_form: str = Field(min_length=1)
    auxiliary_targets: list[AuxiliaryTargetManifest] = Field(default_factory=list)
    data: DataManifest = Field(default_factory=DataManifest)
    training: TrainingManifest
    repair: RepairLineage | None = None


_ALLOWED_FILENAMES = {"manifest.yaml", "model.py", "README.md", "PROPOSAL.md"}
_ALLOWED_SUFFIXES = {".py"}
_FORBIDDEN_SUFFIXES = {
    ".sh",
    ".ipynb",
    ".pt",
    ".pth",
    ".ckpt",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
    ".npy",
    ".npz",
}

_REQUIRED_PROPOSAL_SECTION_TITLES = (
    "hypothesis",
    "comparison target",
    "expected effect",
    "implementation sketch",
    "contract features used",
    "budget requested",
    "success criteria",
    "fallback next decision",
)

_SECTION_SYNONYMS = {
    "hypothesis": {"hypothesis"},
    "comparison target": {"comparison target"},
    "expected effect": {"expected effect"},
    "implementation sketch": {"implementation sketch"},
    "contract features used": {"contract features used", "contract features"},
    "budget requested": {"budget requested", "budget"},
    "success criteria": {"success criteria"},
    "fallback next decision": {"fallback next decision", "fallback decision"},
}

_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(?P<heading>.+?)\s*$")
_METADATA_RE = re.compile(r"^\s*([^:#\n][^:]*)\s*:\s*(.+?)\s*$")


def validate_candidate_directory(
    candidate_dir: str | Path,
    *,
    require_proposal: bool = False,
    require_readme: bool = False,
    research_problem_registry: ResearchProblemSpecRegistry | None = None,
) -> CandidateManifest:
    """Validate a local Candidate Experiment directory and return its manifest.

    Issue 1 validates only the source contract. It does not import or execute
    candidate Python code.

    Args:
        require_proposal: Require a local ``PROPOSAL.md`` file that contains the
            required pre-code Experiment Proposal sections.
        require_readme: Require a local ``README.md`` file for autonomous
            submission documentation.
        research_problem_registry: Trusted registry used to validate
            Research Problem-scoped manifest allowlists. Defaults to built-ins.
    """

    path = Path(candidate_dir)
    if not path.exists():
        raise CandidateValidationError(f"candidate directory does not exist: {path}")
    if not path.is_dir():
        raise CandidateValidationError(f"candidate source must be a directory: {path}")

    _validate_required_files(path)
    _validate_file_allowlist(path)
    if require_proposal:
        _validate_proposal_file(path / "PROPOSAL.md")
    if require_readme:
        _validate_readme_file(path / "README.md")
    return _load_manifest(path / "manifest.yaml", research_problem_registry=research_problem_registry)


def _validate_required_files(path: Path) -> None:
    missing = [name for name in ("manifest.yaml", "model.py") if not (path / name).is_file()]
    if missing:
        raise CandidateValidationError("missing required candidate files: " + ", ".join(missing))


def _validate_file_allowlist(path: Path) -> None:
    for item in path.rglob("*"):
        relative = item.relative_to(path)
        if item.is_symlink():
            raise CandidateValidationError(f"symlink is forbidden in candidate source: {relative}")
        if item.is_dir():
            if item.name.startswith("."):
                raise CandidateValidationError(f"hidden directory is forbidden in candidate source: {relative}")
            continue
        if item.name.startswith("."):
            raise CandidateValidationError(f"hidden file is forbidden in candidate source: {relative}")
        if item.name in _ALLOWED_FILENAMES:
            continue
        if item.suffix in _FORBIDDEN_SUFFIXES:
            raise CandidateValidationError(f"forbidden candidate file: {relative}")
        if item.suffix in _ALLOWED_SUFFIXES:
            continue
        raise CandidateValidationError(f"forbidden candidate file: {relative}")


def _validate_readme_file(path: Path) -> None:
    if not path.is_file():
        raise CandidateValidationError("autonomous-mode requires a candidate-local README.md")


def _validate_proposal_file(path: Path) -> None:
    if not path.is_file():
        raise CandidateValidationError("autonomous-mode requires a candidate-local PROPOSAL.md")

    try:
        text = path.read_text()
    except OSError as exc:
        raise CandidateValidationError(f"cannot read PROPOSAL.md: {exc}") from exc

    found = _proposal_sections_present(text)
    missing = [section for section in _REQUIRED_PROPOSAL_SECTION_TITLES if section not in found]
    if missing:
        raise CandidateValidationError(
            "PROPOSAL.md is missing required sections or metadata: " + ", ".join(_humanize_section(section) for section in missing)
        )


def _normalize_section_name(value: str) -> str:
    tokenized = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(tokenized.split())


def _humanize_section(section: str) -> str:
    return section.replace("/", " ").replace("_", " ").replace("  ", " ").title()


def _proposal_sections_present(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        metadata_match = _METADATA_RE.match(line)

        if heading_match:
            found.update(_match_required_sections(heading_match.group("heading")))
        if metadata_match:
            key, value = metadata_match.groups()
            if value.strip():
                found.update(_match_required_sections(key))

    return found


def _match_required_sections(text: str) -> set[str]:
    normalized = _normalize_section_name(text)
    return {required for required, synonyms in _SECTION_SYNONYMS.items() if normalized in synonyms}


def _load_manifest(
    path: Path,
    *,
    research_problem_registry: ResearchProblemSpecRegistry | None = None,
) -> CandidateManifest:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise CandidateValidationError(f"invalid manifest.yaml: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise CandidateValidationError("manifest.yaml must contain a mapping")

    try:
        manifest = CandidateManifest.model_validate(loaded)
    except ValidationError as exc:
        details = _validation_error_details(exc)
        raise CandidateValidationError("invalid manifest.yaml: " + "; ".join(details)) from exc

    spec = _resolve_research_problem_spec(manifest.research_problem, research_problem_registry)
    details = _manifest_allowlist_errors(manifest, spec)
    if details:
        raise CandidateValidationError("invalid manifest.yaml: " + "; ".join(details))
    return manifest


def _validation_error_details(exc: ValidationError) -> list[str]:
    details = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        input_value = error.get("input")
        input_detail = f" (got {input_value!r})" if input_value is not None else ""
        details.append(f"{loc}: {error['msg']}{input_detail}")
    return details


def _resolve_research_problem_spec(
    spec_id: str,
    registry: ResearchProblemSpecRegistry | None,
) -> ResearchProblemSpec:
    if registry is None:
        if spec_id == DEFAULT_RESEARCH_PROBLEM_ID:
            return get_default_research_problem_spec()
        registry = ResearchProblemSpecRegistry((get_default_research_problem_spec(),))
    try:
        return registry.get(spec_id)
    except UnknownResearchProblemSpecError as exc:
        raise CandidateValidationError(f"invalid manifest.yaml: research_problem: {exc}") from exc


def _manifest_allowlist_errors(manifest: CandidateManifest, spec: ResearchProblemSpec) -> list[str]:
    context = f"for Research Problem {spec.id!r}"
    errors: list[str] = []
    _append_allowlist_error(errors, "input_mode", manifest.input_mode, spec.input_modes, context)
    _append_allowlist_error(errors, "output_form", manifest.output_form, spec.output_forms, context)
    _append_allowlist_error(errors, "training.loss", manifest.training.loss, spec.losses, context)
    _append_allowlist_error(errors, "training.optimizer", manifest.training.optimizer, spec.optimizers, context)
    _append_allowlist_error(errors, "data.sampling_policy", manifest.data.sampling_policy, spec.sampling_policies, context)
    _append_allowlist_error(
        errors,
        "data.augmentation_policy",
        manifest.data.augmentation_policy,
        spec.augmentation_policies,
        context,
    )

    for index, target in enumerate(manifest.auxiliary_targets):
        prefix = f"auxiliary_targets.{index}"
        _append_allowlist_error(errors, f"{prefix}.name", target.name, spec.auxiliary_targets, context)
        _append_allowlist_error(errors, f"{prefix}.loss", target.loss, spec.auxiliary_losses, context)
        expected_output = spec.auxiliary_outputs.get(target.name)
        if expected_output is None:
            # The target-name error above is clearer than also reporting output mismatch.
            continue
        if target.output != expected_output:
            errors.append(f"{prefix}.output: {target.name} auxiliary target must use {expected_output} {context} (got {target.output!r})")
    return errors


def _append_allowlist_error(
    errors: list[str],
    field: str,
    value: str,
    allowed: tuple[str, ...],
    context: str,
) -> None:
    if value not in allowed:
        allowed_values = ", ".join(repr(item) for item in allowed)
        errors.append(f"{field}: unsupported value {context}; expected one of {allowed_values} (got {value!r})")
