"""In-container operation runner for Docker-backed Candidate Execution Boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ml_autoresearch.errors import HARNESS_FAILURE_MARKER, HarnessBootstrapError, ResearchProblemDataError
from ml_autoresearch.evaluations import DEFAULT_MAX_ARTIFACT_SAMPLES
from ml_autoresearch.operations import OperationRequest, execute_operation_request
from ml_autoresearch.research_problems import ResearchProblemProviderConfig, ResearchProblemProviderLoadError


def _is_harness_failure(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, (HarnessBootstrapError, ResearchProblemDataError, ResearchProblemProviderLoadError)):
            return True
        if getattr(current, "failure_classification", None) == "harness_failure":
            return True
        current = current.__cause__
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed ML Autoresearch container operations.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run_operation = subparsers.add_parser("run-operation")
    run_operation.add_argument("--request-json", required=True)
    train_synthetic = subparsers.add_parser("train-synthetic")
    train_synthetic.add_argument("--max-prediction-samples", type=int, default=2)
    train_synthetic.add_argument("--prediction-sample-policy", choices=["first_n", "adjacent_and_scattered"], default="first_n")
    train_research_problem_parser = subparsers.add_parser("train-research-problem")
    train_research_problem_parser.add_argument("--provider-config-json", required=True)
    train_research_problem_parser.add_argument("--max-samples", type=int, default=None)
    train_research_problem_parser.add_argument("--max-prediction-samples", type=int, default=2)
    train_research_problem_parser.add_argument(
        "--prediction-sample-policy", choices=["first_n", "adjacent_and_scattered"], default="first_n"
    )
    evaluate_run_parser = subparsers.add_parser("evaluate-run")
    evaluate_run_parser.add_argument("--data-root", type=Path, default=Path("/data"))
    evaluate_run_parser.add_argument("--max-artifact-samples", type=int, default=DEFAULT_MAX_ARTIFACT_SAMPLES)
    evaluate_run_parser.add_argument("--backend", choices=["native"], default="native")
    post_run_parser = subparsers.add_parser("run-post-run-evaluation")
    post_run_parser.add_argument("--request", type=Path, required=True)
    post_run_parser.add_argument("--runs-root", type=Path, required=True)
    post_run_parser.add_argument("--ledger-path", type=Path, required=True)
    args = parser.parse_args()

    if args.operation == "run-operation":
        try:
            execute_operation_request(OperationRequest.from_json(args.request_json))
        except Exception as exc:  # noqa: BLE001 - emit a stable trusted-boundary classification marker.
            if _is_harness_failure(exc):
                print(HARNESS_FAILURE_MARKER, file=sys.stderr)
            raise
    elif args.operation == "train-synthetic":
        execute_operation_request(
            OperationRequest(
                operation="train_synthetic",
                run_dir=Path("/"),
                max_prediction_samples=args.max_prediction_samples,
                prediction_sample_policy=args.prediction_sample_policy,
            )
        )
    elif args.operation == "train-research-problem":
        provider_config = ResearchProblemProviderConfig.model_validate_json(args.provider_config_json)
        execute_operation_request(
            OperationRequest(
                operation="train_research_problem",
                run_dir=Path("/"),
                provider_config=provider_config,
                max_samples=args.max_samples,
                max_prediction_samples=args.max_prediction_samples,
                prediction_sample_policy=args.prediction_sample_policy,
            )
        )
    elif args.operation == "evaluate-run":
        execute_operation_request(
            OperationRequest(
                operation="evaluate_run",
                run_dir=Path("/"),
                data_root=args.data_root,
                max_artifact_samples=args.max_artifact_samples,
            )
        )
    elif args.operation == "run-post-run-evaluation":
        execute_operation_request(
            OperationRequest(
                operation="run_post_run_evaluation",
                request_path=args.request,
                runs_root=args.runs_root,
                ledger_path=args.ledger_path,
            )
        )


if __name__ == "__main__":
    main()
