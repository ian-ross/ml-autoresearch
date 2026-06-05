"""Trusted Research Problem Spec registry.

The registry is a Harness-owned seam for declarative Research Problem policy. It
is not a Candidate Experiment extension point; Candidate Experiments still use
only the existing allowlisted manifest contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_RESEARCH_PROBLEM_ID = "ground_camera_contrail_detection"


class ResearchProblemSpecError(ValueError):
    """Base error for Research Problem Spec registry failures."""


class DuplicateResearchProblemSpecError(ResearchProblemSpecError):
    """Raised when registering two specs with the same id."""


class UnknownResearchProblemSpecError(ResearchProblemSpecError):
    """Raised when a spec id is not present in a registry."""


class ResearchProblemSpec(BaseModel):
    """Declarative Harness-owned capabilities for one Research Problem.

    This first representation intentionally contains only stable allowlisted
    contract choices. Executable behavior such as data adapters, metric
    functions, and target construction remains in existing Harness code until
    downstream callers are deliberately migrated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
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
    augmentation_policies: tuple[str, ...] = Field(min_length=1)
    primary_metric: str = Field(min_length=1)

    @model_validator(mode="after")
    def spec_mappings_match_allowlists(self) -> "ResearchProblemSpec":
        unknown_inputs = set(self.input_specs) - set(self.input_modes)
        if unknown_inputs:
            unknown = ", ".join(sorted(unknown_inputs))
            raise ValueError(f"input_specs contains unknown input mode(s): {unknown}")
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


class ResearchProblemSpecRegistry:
    """In-memory registry for trusted Research Problem Specs."""

    def __init__(
        self,
        specs: Iterable[ResearchProblemSpec] = (),
        *,
        default_id: str = DEFAULT_RESEARCH_PROBLEM_ID,
    ) -> None:
        self._specs: dict[str, ResearchProblemSpec] = {}
        self._default_id = default_id
        for spec in specs:
            self.register(spec)

    def register(self, spec: ResearchProblemSpec) -> ResearchProblemSpec:
        """Register a spec and return it.

        Duplicate ids are rejected so lookup by id stays unambiguous.
        """

        if spec.id in self._specs:
            raise DuplicateResearchProblemSpecError(f"research problem spec already registered: {spec.id}")
        self._specs[spec.id] = spec
        return spec

    def get(self, spec_id: str) -> ResearchProblemSpec:
        """Return the registered Research Problem Spec for ``spec_id``."""

        try:
            return self._specs[spec_id]
        except KeyError as exc:
            raise UnknownResearchProblemSpecError(f"unknown research problem spec: {spec_id}") from exc

    def default(self) -> ResearchProblemSpec:
        """Return the default Research Problem Spec."""

        return self.get(self._default_id)

    def ids(self) -> tuple[str, ...]:
        """Return registered Research Problem ids in deterministic order."""

        return tuple(sorted(self._specs))


GROUND_CAMERA_CONTRAIL_DETECTION_SPEC = ResearchProblemSpec(
    id=DEFAULT_RESEARCH_PROBLEM_ID,
    version="v0",
    input_modes=("single_frame_rgb",),
    input_specs={"single_frame_rgb": {"mode": "single_frame_rgb", "shape": [3, 128, 128]}},
    output_forms=("mask_logits",),
    output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 128, 128]}},
    auxiliary_targets=("line", "boundary"),
    auxiliary_outputs={"line": "line_logits", "boundary": "boundary_logits"},
    auxiliary_output_shapes={"line": [1, 128, 128], "boundary": [1, 128, 128]},
    losses=("bce_dice",),
    auxiliary_losses=("weighted_bce",),
    optimizers=("adamw",),
    sampling_policies=("sequential", "deterministic_shuffle"),
    augmentation_policies=(
        "none",
        "light_geometric",
        "light_photometric",
        "light_combined",
    ),
    primary_metric="val/dice",
)


_BUILTIN_REGISTRY = ResearchProblemSpecRegistry(
    (GROUND_CAMERA_CONTRAIL_DETECTION_SPEC,), default_id=DEFAULT_RESEARCH_PROBLEM_ID
)


def get_research_problem_spec(spec_id: str) -> ResearchProblemSpec:
    """Look up a built-in Research Problem Spec by id."""

    return _BUILTIN_REGISTRY.get(spec_id)


def get_default_research_problem_spec() -> ResearchProblemSpec:
    """Return the default built-in Research Problem Spec."""

    return _BUILTIN_REGISTRY.default()


def registered_research_problem_ids() -> tuple[str, ...]:
    """Return ids for built-in registered Research Problem Specs."""

    return _BUILTIN_REGISTRY.ids()
