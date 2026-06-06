"""Trusted Research Problem Spec registry.

The registry is a Harness-owned seam for declarative Research Problem policy. It
is not a Candidate Experiment extension point; Candidate Experiments still use
only the existing allowlisted manifest contract.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


_SUBPROCESS_RUN = subprocess.run


class ResearchProblemSpecError(ValueError):
    """Base error for Research Problem Spec registry failures."""


class ResearchProblemProviderLoadError(ResearchProblemSpecError):
    """Raised when a configured filesystem Research Problem provider cannot be loaded."""


class DuplicateResearchProblemSpecError(ResearchProblemSpecError):
    """Raised when registering two specs with the same id."""


class UnknownResearchProblemSpecError(ResearchProblemSpecError):
    """Raised when a spec id is not present in a registry."""


class ResearchProblemBriefDocument(BaseModel):
    """Advisory Research Problem Brief document declared by a provider."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    path: Path
    summary: str | None = None
    required: bool = False

    @field_validator("path")
    @classmethod
    def path_must_be_provider_relative(cls, path: Path) -> Path:
        path_text = str(path)
        if path.is_absolute():
            raise ValueError("brief document path must be relative to the Research Problem package")
        if "\\" in path_text:
            raise ValueError("brief document path must use portable forward-slash separators")
        if any(part in ("", ".", "..") for part in path.parts):
            raise ValueError("brief document path must not contain empty, current-directory, or parent-directory parts")
        return path


class ResolvedResearchProblemBriefDocument(BaseModel):
    """Research Problem Brief document with its provider-root-relative path resolved."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    path: Path
    resolved_path: Path
    summary: str | None = None
    required: bool = False

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "name": self.name,
            "role": self.role,
            "path": self.path.as_posix(),
            "resolved_path": str(self.resolved_path),
            "required": self.required,
        }
        if self.summary is not None:
            metadata["summary"] = self.summary
        return metadata


class ResearchProblemSpec(BaseModel):
    """Declarative Harness-owned capabilities for one Research Problem.

    This first representation intentionally contains only stable allowlisted
    contract choices. Executable behavior such as data adapters, metric
    functions, and target construction remains in existing Harness code until
    downstream callers are deliberately migrated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    contract_version: str = Field(default="v0", min_length=1)
    input_modes: tuple[str, ...] = Field(min_length=1)
    input_specs: dict[str, dict[str, object]] = Field(default_factory=dict)
    output_forms: tuple[str, ...] = Field(min_length=1)
    output_specs: dict[str, dict[str, object]] = Field(default_factory=dict)
    auxiliary_targets: tuple[str, ...] = ()
    auxiliary_outputs: dict[str, str] = Field(default_factory=dict)
    auxiliary_output_shapes: dict[str, list[int]] = Field(default_factory=dict)
    losses: tuple[str, ...] = Field(min_length=1)
    auxiliary_losses: tuple[str, ...] = ()
    optimizers: tuple[str, ...] = Field(min_length=1)
    sampling_policies: tuple[str, ...] = Field(min_length=1)
    frame_selection_policies: tuple[str, ...] = ("all_target_frames",)
    input_mode_frame_selection_defaults: dict[str, str] = Field(default_factory=dict)
    augmentation_policies: tuple[str, ...] = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    brief_documents: tuple[ResearchProblemBriefDocument, ...] = ()
    training_adapter: object | None = Field(default=None, exclude=True)
    evaluation_adapter: object | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def spec_mappings_match_allowlists(self) -> "ResearchProblemSpec":
        unknown_inputs = set(self.input_specs) - set(self.input_modes)
        if unknown_inputs:
            unknown = ", ".join(sorted(unknown_inputs))
            raise ValueError(f"input_specs contains unknown input mode(s): {unknown}")
        unknown_frame_selection_defaults = set(self.input_mode_frame_selection_defaults) - set(self.input_modes)
        if unknown_frame_selection_defaults:
            unknown = ", ".join(sorted(unknown_frame_selection_defaults))
            raise ValueError(f"input_mode_frame_selection_defaults contains unknown input mode(s): {unknown}")
        unknown_frame_selection_policies = set(self.input_mode_frame_selection_defaults.values()) - set(self.frame_selection_policies)
        if unknown_frame_selection_policies:
            unknown = ", ".join(sorted(unknown_frame_selection_policies))
            raise ValueError(f"input_mode_frame_selection_defaults contains unknown frame selection policy/policies: {unknown}")
        unknown_outputs = set(self.output_specs) - set(self.output_forms)
        if unknown_outputs:
            unknown = ", ".join(sorted(unknown_outputs))
            raise ValueError(f"output_specs contains unknown output form(s): {unknown}")
        unknown_auxiliary_outputs = set(self.auxiliary_outputs) - set(self.auxiliary_targets)
        if unknown_auxiliary_outputs:
            unknown = ", ".join(sorted(unknown_auxiliary_outputs))
            raise ValueError(f"auxiliary_outputs contains unknown target(s): {unknown}")
        unknown_auxiliary_shapes = set(self.auxiliary_output_shapes) - set(self.auxiliary_targets)
        if unknown_auxiliary_shapes:
            unknown = ", ".join(sorted(unknown_auxiliary_shapes))
            raise ValueError(f"auxiliary_output_shapes contains unknown target(s): {unknown}")
        brief_names = [document.name for document in self.brief_documents]
        duplicate_brief_names = {name for name in brief_names if brief_names.count(name) > 1}
        if duplicate_brief_names:
            duplicates = ", ".join(sorted(duplicate_brief_names))
            raise ValueError(f"brief_documents contains duplicate document name(s): {duplicates}")
        return self

    def build_input_spec(self, resolved_manifest: Mapping[str, object]) -> dict[str, object]:
        """Build the smoke-test input spec for a Resolved Manifest."""

        input_mode = str(resolved_manifest.get("input_mode", ""))
        try:
            return dict(self.input_specs[input_mode])
        except KeyError as exc:
            raise ResearchProblemSpecError(
                f"Research Problem {self.id!r} has no smoke input spec for input_mode {input_mode!r}"
            ) from exc

    def build_output_spec(self, resolved_manifest: Mapping[str, object]) -> dict[str, object]:
        """Build the smoke-test output spec for a Resolved Manifest."""

        output_form = str(resolved_manifest.get("output_form", ""))
        try:
            output_spec = dict(self.output_specs[output_form])
        except KeyError as exc:
            raise ResearchProblemSpecError(
                f"Research Problem {self.id!r} has no smoke output spec for output_form {output_form!r}"
            ) from exc

        auxiliary_outputs = []
        for target in resolved_manifest.get("auxiliary_targets", []) or []:
            if not isinstance(target, Mapping):
                raise ResearchProblemSpecError("resolved manifest auxiliary_targets must contain mappings")
            target_name = str(target.get("name", ""))
            try:
                output_name = self.auxiliary_outputs[target_name]
                shape = self.auxiliary_output_shapes[target_name]
            except KeyError as exc:
                raise ResearchProblemSpecError(
                    f"Research Problem {self.id!r} has no smoke auxiliary output spec for target {target_name!r}"
                ) from exc
            auxiliary_outputs.append({"target": target_name, "name": output_name, "shape": list(shape)})
        if auxiliary_outputs:
            output_spec["auxiliary_outputs"] = auxiliary_outputs
        return output_spec


class ResearchProblemProviderConfig(BaseModel):
    """Harness-owned configuration for a trusted filesystem Research Problem provider."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str = Field(min_length=1)
    package_root: Path
    provider_target: str = Field(min_length=1)
    expected_contract_version: str = Field(min_length=1)
    data_config: dict[str, object] = Field(default_factory=dict)


class ResearchProblemProviderProvenance(BaseModel):
    """Source provenance for a loaded filesystem Research Problem provider."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    resolved_package_root: Path
    provider_target: str = Field(min_length=1)
    git: dict[str, object] | None = None

    def run_metadata(self) -> dict[str, object]:
        """Return JSON-serialisable provider provenance for Run metadata."""

        metadata: dict[str, object] = {
            "target": self.provider_target,
            "resolved_package_root": str(self.resolved_package_root),
        }
        if self.git is not None:
            metadata["git"] = self.git
        return metadata


class LoadedResearchProblemSpec(BaseModel):
    """A checked Research Problem Spec plus provider provenance."""

    model_config = ConfigDict(frozen=True)

    spec: ResearchProblemSpec
    provenance: ResearchProblemProviderProvenance
    brief_documents: tuple[ResolvedResearchProblemBriefDocument, ...] = ()

    def run_metadata(self) -> dict[str, object]:
        """Return JSON-serialisable Research Problem metadata for Run recording."""

        metadata: dict[str, object] = {
            "id": self.spec.id,
            "version": self.spec.version,
            "contract_version": self.spec.contract_version,
            "provider": self.provenance.run_metadata(),
        }
        if self.brief_documents:
            metadata["brief_documents"] = [document.metadata() for document in self.brief_documents]
        return metadata


class ResearchProblemSpecRegistry:
    """In-memory registry for trusted Research Problem Specs."""

    def __init__(
        self,
        specs: Iterable[ResearchProblemSpec] = (),
        *,
        active_id: str | None = None,
    ) -> None:
        self._specs: dict[str, ResearchProblemSpec] = {}
        self._provenance: dict[str, ResearchProblemProviderProvenance] = {}
        self._active_id = active_id
        for spec in specs:
            self.register(spec)

    def register(
        self,
        spec: ResearchProblemSpec,
        *,
        provenance: ResearchProblemProviderProvenance | None = None,
    ) -> ResearchProblemSpec:
        """Register a spec and return it.

        Duplicate ids are rejected so lookup by id stays unambiguous.
        """

        if spec.id in self._specs:
            raise DuplicateResearchProblemSpecError(f"research problem spec already registered: {spec.id}")
        self._specs[spec.id] = spec
        if provenance is not None:
            self._provenance[spec.id] = provenance
        return spec

    def get(self, spec_id: str) -> ResearchProblemSpec:
        """Return the registered Research Problem Spec for ``spec_id``."""

        try:
            return self._specs[spec_id]
        except KeyError as exc:
            raise UnknownResearchProblemSpecError(f"unknown research problem spec: {spec_id}") from exc

    def active(self) -> ResearchProblemSpec:
        """Return the explicitly configured active Research Problem Spec."""

        if self._active_id is None:
            raise UnknownResearchProblemSpecError("no active research problem spec configured")
        return self.get(self._active_id)

    def ids(self) -> tuple[str, ...]:
        """Return registered Research Problem ids in deterministic order."""

        return tuple(sorted(self._specs))

    def get_provenance(self, spec_id: str) -> ResearchProblemProviderProvenance | None:
        """Return provider provenance for ``spec_id`` when it was loaded from a provider."""

        self.get(spec_id)
        return self._provenance.get(spec_id)


def load_research_problem_provider(
    config: ResearchProblemProviderConfig,
    *,
    registry: ResearchProblemSpecRegistry | None = None,
) -> LoadedResearchProblemSpec:
    """Load, validate, and optionally register a trusted filesystem Research Problem provider."""

    package_root = _resolve_package_root(config.package_root)
    module_name, symbol_name = _parse_provider_target(config.provider_target)
    provider = _import_provider(package_root, module_name, symbol_name)
    raw_spec = _call_provider(provider, config)
    spec = _check_provider_spec(raw_spec, config)
    provenance = ResearchProblemProviderProvenance(
        resolved_package_root=package_root,
        provider_target=config.provider_target,
        git=_git_provenance(package_root),
    )
    brief_documents = _resolve_brief_documents(spec, package_root)
    if registry is not None:
        registry.register(spec, provenance=provenance)
    return LoadedResearchProblemSpec(spec=spec, provenance=provenance, brief_documents=brief_documents)


def _resolve_package_root(package_root: Path) -> Path:
    try:
        resolved = package_root.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchProblemProviderLoadError(f"Research Problem package root does not exist: {package_root}") from exc
    if not resolved.is_dir():
        raise ResearchProblemProviderLoadError(f"Research Problem package root is not a directory: {resolved}")
    return resolved


def _parse_provider_target(provider_target: str) -> tuple[str, str]:
    if ":" not in provider_target:
        raise ResearchProblemProviderLoadError(
            f"Research Problem provider target must be 'module:symbol', got {provider_target!r}"
        )
    module_name, symbol_name = provider_target.split(":", 1)
    if not module_name or not symbol_name:
        raise ResearchProblemProviderLoadError(
            f"Research Problem provider target must be 'module:symbol', got {provider_target!r}"
        )
    return module_name, symbol_name


def _import_provider(package_root: Path, module_name: str, symbol_name: str) -> Callable[..., object]:
    package_root_text = str(package_root)
    inserted = False
    if package_root_text not in sys.path:
        sys.path.insert(0, package_root_text)
        inserted = True
    try:
        _evict_provider_module_cache(package_root, module_name)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ResearchProblemProviderLoadError(
                f"cannot import Research Problem provider module {module_name!r} from {package_root}"
            ) from exc
        try:
            provider = getattr(module, symbol_name)
        except AttributeError as exc:
            raise ResearchProblemProviderLoadError(
                f"Research Problem provider module {module_name!r} does not define provider symbol {symbol_name!r}"
            ) from exc
        if not callable(provider):
            raise ResearchProblemProviderLoadError(
                f"Research Problem provider target {module_name}:{symbol_name} is not callable"
            )
        return provider
    finally:
        if inserted:
            try:
                sys.path.remove(package_root_text)
            except ValueError:
                pass


def _evict_provider_module_cache(package_root: Path, module_name: str) -> None:
    """Avoid reusing a same-named provider module from another filesystem package."""

    names = [module_name]
    top_level = module_name.split(".", 1)[0]
    if top_level != module_name:
        names.append(top_level)
    for name in names:
        cached = sys.modules.get(name)
        cached_file = getattr(cached, "__file__", None) if cached is not None else None
        if cached_file is None:
            continue
        try:
            cached_path = Path(cached_file).resolve()
            cached_path.relative_to(package_root)
        except (OSError, ValueError):
            sys.modules.pop(name, None)


def _call_provider(provider: Callable[..., object], config: ResearchProblemProviderConfig) -> object:
    try:
        return provider(data_config=dict(config.data_config))
    except TypeError:
        try:
            return provider()
        except Exception as exc:
            raise ResearchProblemProviderLoadError(f"Research Problem provider {config.provider_target!r} failed: {exc}") from exc
    except Exception as exc:
        raise ResearchProblemProviderLoadError(f"Research Problem provider {config.provider_target!r} failed: {exc}") from exc


def _check_provider_spec(raw_spec: object, config: ResearchProblemProviderConfig) -> ResearchProblemSpec:
    if isinstance(raw_spec, ResearchProblemSpec):
        spec = raw_spec
    else:
        try:
            spec = ResearchProblemSpec.model_validate(raw_spec)
        except ValidationError as exc:
            raise ResearchProblemProviderLoadError(f"invalid Research Problem Spec returned by provider: {exc}") from exc
    if spec.id != config.id:
        raise ResearchProblemProviderLoadError(
            f"Research Problem Spec id mismatch: configured {config.id!r}, provider returned {spec.id!r}"
        )
    if spec.contract_version != config.expected_contract_version:
        raise ResearchProblemProviderLoadError(
            "Research Problem Spec contract-version mismatch: "
            f"configured {config.expected_contract_version!r}, provider returned {spec.contract_version!r}"
        )
    return spec


def _resolve_brief_documents(
    spec: ResearchProblemSpec,
    package_root: Path,
) -> tuple[ResolvedResearchProblemBriefDocument, ...]:
    resolved_documents: list[ResolvedResearchProblemBriefDocument] = []
    for document in spec.brief_documents:
        try:
            resolved_path = (package_root / document.path).resolve(strict=False)
            resolved_path.relative_to(package_root)
        except (OSError, ValueError) as exc:
            raise ResearchProblemProviderLoadError(
                f"Research Problem Brief document {document.name!r} path escapes package root: {document.path}"
            ) from exc
        if document.required and not resolved_path.is_file():
            raise ResearchProblemProviderLoadError(
                f"required Research Problem Brief document {document.name!r} does not exist: {document.path}"
            )
        resolved_documents.append(
            ResolvedResearchProblemBriefDocument(
                name=document.name,
                role=document.role,
                path=document.path,
                resolved_path=resolved_path,
                summary=document.summary,
                required=document.required,
            )
        )
    return tuple(resolved_documents)


def _git_provenance(package_root: Path) -> dict[str, object] | None:
    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return _SUBPROCESS_RUN(
            ["git", "-C", str(package_root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    inside = run_git("rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    commit = run_git("rev-parse", "HEAD")
    if commit.returncode != 0:
        return None
    status = run_git("status", "--porcelain")
    dirty = status.returncode == 0 and bool(status.stdout.strip())
    return {"commit": commit.stdout.strip(), "dirty": dirty}


