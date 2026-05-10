"""Validated Evaluation Request workflow for Harness-owned Post-Run Evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER, record_research_event


class EvaluationRequestError(ValueError):
    """Raised when an Evaluation Request cannot be validated or executed."""


EvaluationMode = Literal["threshold_sweep", "failure_bucket_review"]


class ThresholdSweep(BaseModel):
    """Bounded probability-threshold sweep parameters."""

    model_config = ConfigDict(extra="forbid")

    min: float = Field(ge=0.0, le=1.0)
    max: float = Field(ge=0.0, le=1.0)
    steps: int = Field(ge=2, le=101)

    @model_validator(mode="after")
    def _bounds_in_order(self) -> "ThresholdSweep":
        if self.min >= self.max:
            raise ValueError("threshold_sweep min must be less than max")
        return self


class DiagnosticParameters(BaseModel):
    """Harness-validated bounded parameters for approved Post-Run Evaluations."""

    model_config = ConfigDict(extra="forbid")

    primary_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_sweep: ThresholdSweep | None = None
    batch_size: int | None = Field(default=None, ge=1, le=1024)
    artifact_count: int | None = Field(default=None, ge=0, le=100)
    failure_bucket_count: int | None = Field(default=None, ge=1, le=20)


class ArtifactBudget(BaseModel):
    """Resource budget enforced before the Harness writes evaluation artifacts."""

    model_config = ConfigDict(extra="forbid")

    max_artifacts: int = Field(ge=0, le=100)
    max_runtime_seconds: int = Field(ge=1, le=3600)


class EvaluationRequest(BaseModel):
    """Auditable authorization to run one bounded Harness-owned Post-Run Evaluation."""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, min_length=1)
    target_run_id: str = Field(min_length=1)
    evaluation_mode: EvaluationMode
    diagnostic_question: str = Field(min_length=1)
    expected_decision_impact: str = Field(min_length=1)
    parameters: DiagnosticParameters = Field(default_factory=DiagnosticParameters)
    artifact_budget: ArtifactBudget

    @model_validator(mode="after")
    def _mode_parameters_are_coherent(self) -> "EvaluationRequest":
        if self.evaluation_mode == "threshold_sweep" and self.parameters.threshold_sweep is None:
            raise ValueError("threshold_sweep evaluation requires parameters.threshold_sweep")
        if self.evaluation_mode == "failure_bucket_review" and self.parameters.failure_bucket_count is None:
            raise ValueError("failure_bucket_review evaluation requires parameters.failure_bucket_count")
        if self.parameters.artifact_count is not None and self.parameters.artifact_count > self.artifact_budget.max_artifacts:
            raise ValueError("parameters.artifact_count must not exceed artifact_budget.max_artifacts")
        return self


def validate_evaluation_request_file(request_path: str | Path) -> EvaluationRequest:
    """Load and validate one YAML Evaluation Request file."""

    path = Path(request_path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise EvaluationRequestError(f"cannot read Evaluation Request {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise EvaluationRequestError(f"invalid Evaluation Request YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EvaluationRequestError("Evaluation Request must be a YAML mapping")
    try:
        request = EvaluationRequest.model_validate(raw)
    except ValidationError as exc:
        raise EvaluationRequestError(str(exc)) from exc
    if request.request_id is None:
        request = request.model_copy(update={"request_id": path.stem})
    return request


def run_post_run_evaluation(
    request_path: str | Path,
    *,
    runs_root: str | Path,
    ledger_path: str | Path = CANONICAL_RESEARCH_LEDGER,
) -> dict[str, Any]:
    """Run one approved autonomous Post-Run Evaluation from a validated request.

    The request is validated before any artifact or ledger write occurs, so an
    invalid mode or unbounded parameter cannot create a misleading successful
    evaluation record.
    """

    path = Path(request_path)
    request = validate_evaluation_request_file(path)
    assert request.request_id is not None
    root = Path(runs_root)
    run_dir = root / request.target_run_id
    if not run_dir.is_dir():
        raise EvaluationRequestError(f"target Run does not exist: {request.target_run_id}")
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise EvaluationRequestError(f"target Run is missing run_metadata.json: {request.target_run_id}")

    evaluation_id = _evaluation_id(request.request_id)
    relative_evaluation_dir = Path("evaluations") / evaluation_id
    evaluation_dir = run_dir / relative_evaluation_dir
    if evaluation_dir.exists():
        raise EvaluationRequestError(f"evaluation already exists: {evaluation_id}")

    requested_event = record_research_event(
        "evaluation_requested",
        {
            "evaluation_request_id": request.request_id,
            "request_path": str(path),
            "run_id": request.target_run_id,
            "evaluation_mode": request.evaluation_mode,
        },
        ledger_path=ledger_path,
    )

    evaluation_dir.mkdir(parents=True)
    summary_rel = relative_evaluation_dir / "summary.json"
    metadata_rel = relative_evaluation_dir / "evaluation_metadata.json"
    summary = _build_summary(request, evaluation_id)
    metadata = {
        "evaluation_id": evaluation_id,
        "request_id": request.request_id,
        "parent_run_id": request.target_run_id,
        "evaluation_mode": request.evaluation_mode,
        "diagnostic_question": request.diagnostic_question,
        "expected_decision_impact": request.expected_decision_impact,
        "parameters": request.parameters.model_dump(exclude_none=True),
        "artifact_budget": request.artifact_budget.model_dump(),
        "artifacts": {"summary": str(summary_rel)},
    }
    _write_json(evaluation_dir / "summary.json", summary)
    _write_json(evaluation_dir / "evaluation_metadata.json", metadata)

    completed_event = record_research_event(
        "evaluation_completed",
        {
            "evaluation_id": evaluation_id,
            "evaluation_request_id": request.request_id,
            "run_id": request.target_run_id,
            "evaluation_mode": request.evaluation_mode,
            "artifact_metadata_path": str(Path(root.name) / request.target_run_id / metadata_rel),
        },
        ledger_path=ledger_path,
    )
    return {"request": request.model_dump(), "evaluation": metadata, "ledger_events": [requested_event, completed_event], "evaluation_id": evaluation_id}


def _build_summary(request: EvaluationRequest, evaluation_id: str) -> dict[str, Any]:
    return {
        "evaluation_id": evaluation_id,
        "request_id": request.request_id,
        "parent_run_id": request.target_run_id,
        "evaluation_mode": request.evaluation_mode,
        "diagnostic_question": request.diagnostic_question,
        "expected_decision_impact": request.expected_decision_impact,
        "status": "completed",
        "note": "This Harness-owned Post-Run Evaluation recorded the validated request and artifact linkage; mode-specific metric computation can deepen behind this request gate.",
    }


def _evaluation_id(request_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in request_id)
    return f"eval_{safe}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
