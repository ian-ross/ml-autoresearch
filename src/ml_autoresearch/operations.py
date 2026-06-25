"""Shared Candidate Execution Boundary operation dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json

from ml_autoresearch.research_problems import ResearchProblemProviderConfig

OperationName = Literal[
    "smoke_test",
    "train_synthetic",
    "train_research_problem",
    "evaluate_run",
    "run_post_run_evaluation",
]


@dataclass(frozen=True)
class OperationResponse:
    """Serializable result from a dispatched Harness operation."""

    operation: OperationName
    parameter_count: int | None = None
    input_spec: dict[str, object] | None = None
    output_spec: dict[str, object] | None = None


@dataclass(frozen=True)
class OperationRequest:
    """Serializable request for one Harness operation."""

    operation: OperationName
    run_dir: Path = Path("/")
    provider_config: ResearchProblemProviderConfig | None = None
    max_samples: int | None = None
    max_prediction_samples: int = 2
    prediction_sample_policy: str = "first_n"
    data_root: Path | None = None
    max_artifact_samples: int = 12
    request_path: Path | None = None
    runs_root: Path | None = None
    ledger_path: Path | None = None

    def to_json(self) -> str:
        payload: dict[str, object] = {
            "operation": self.operation,
            "run_dir": str(self.run_dir),
            "max_prediction_samples": self.max_prediction_samples,
            "prediction_sample_policy": self.prediction_sample_policy,
            "max_artifact_samples": self.max_artifact_samples,
        }
        if self.provider_config is not None:
            payload["provider_config"] = self.provider_config.model_dump(mode="json")
        if self.max_samples is not None:
            payload["max_samples"] = self.max_samples
        if self.data_root is not None:
            payload["data_root"] = str(self.data_root)
        if self.request_path is not None:
            payload["request_path"] = str(self.request_path)
        if self.runs_root is not None:
            payload["runs_root"] = str(self.runs_root)
        if self.ledger_path is not None:
            payload["ledger_path"] = str(self.ledger_path)
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "OperationRequest":
        payload = json.loads(value)
        provider_payload = payload.get("provider_config")
        provider_config = (
            ResearchProblemProviderConfig.model_validate(provider_payload) if isinstance(provider_payload, dict) else None
        )
        return cls(
            operation=payload["operation"],
            run_dir=Path(payload.get("run_dir", "/")),
            provider_config=provider_config,
            max_samples=payload.get("max_samples"),
            max_prediction_samples=int(payload.get("max_prediction_samples", 2)),
            prediction_sample_policy=str(payload.get("prediction_sample_policy", "first_n")),
            data_root=Path(payload["data_root"]) if payload.get("data_root") is not None else None,
            max_artifact_samples=int(payload.get("max_artifact_samples", 12)),
            request_path=Path(payload["request_path"]) if payload.get("request_path") is not None else None,
            runs_root=Path(payload["runs_root"]) if payload.get("runs_root") is not None else None,
            ledger_path=Path(payload["ledger_path"]) if payload.get("ledger_path") is not None else None,
        )


def execute_operation_request(request: OperationRequest) -> OperationResponse:
    """Execute one Harness operation through the shared dispatcher."""

    if request.operation == "smoke_test":
        from ml_autoresearch.smoke import smoke_test_run

        result = smoke_test_run(request.run_dir)
        return OperationResponse(
            operation=request.operation,
            parameter_count=result.parameter_count,
            input_spec=result.input_spec,
            output_spec=result.output_spec,
        )
    if request.operation == "train_synthetic":
        from ml_autoresearch.training import train_synthetic_fixture_run

        train_synthetic_fixture_run(
            request.run_dir,
            max_prediction_samples=request.max_prediction_samples,
            prediction_sample_policy=request.prediction_sample_policy,
        )
        return OperationResponse(operation=request.operation)
    if request.operation == "train_research_problem":
        from ml_autoresearch.training import train_research_problem_run

        if request.provider_config is None:
            raise ValueError("train_research_problem requires provider_config")
        train_research_problem_run(
            request.run_dir,
            request.provider_config,
            max_samples=request.max_samples,
            max_prediction_samples=request.max_prediction_samples,
            prediction_sample_policy=request.prediction_sample_policy,
        )
        return OperationResponse(operation=request.operation)
    if request.operation == "evaluate_run":
        from ml_autoresearch.evaluations import evaluate_run

        evaluate_run(
            request.run_dir,
            backend="native",
            data_root=request.data_root,
            max_artifact_samples=request.max_artifact_samples,
        )
        return OperationResponse(operation=request.operation)
    if request.operation == "run_post_run_evaluation":
        from ml_autoresearch.evaluation_requests import run_post_run_evaluation

        if request.request_path is None or request.runs_root is None or request.ledger_path is None:
            raise ValueError("run_post_run_evaluation requires request_path, runs_root, and ledger_path")
        run_post_run_evaluation(request.request_path, runs_root=request.runs_root, ledger_path=request.ledger_path)
        return OperationResponse(operation=request.operation)
    raise ValueError(f"unsupported operation: {request.operation}")
