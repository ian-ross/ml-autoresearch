"""In-container operation runner for Docker-backed Candidate Execution Boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_autoresearch.evaluations import DEFAULT_MAX_ARTIFACT_SAMPLES, evaluate_run
from ml_autoresearch.research_problems import ResearchProblemProviderConfig
from ml_autoresearch.training import train_gvccs, train_research_problem_run, train_synthetic_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed ML Autoresearch container operations.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    train_synthetic = subparsers.add_parser("train-synthetic")
    train_synthetic.add_argument("--max-prediction-samples", type=int, default=2)
    train_synthetic.add_argument("--prediction-sample-policy", choices=["first_n", "adjacent_and_scattered"], default="first_n")
    train_gvccs_parser = subparsers.add_parser("train-gvccs")
    train_gvccs_parser.add_argument("--max-samples", type=int, default=None)
    train_gvccs_parser.add_argument("--max-prediction-samples", type=int, default=2)
    train_gvccs_parser.add_argument("--prediction-sample-policy", choices=["first_n", "adjacent_and_scattered"], default="first_n")
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
    args = parser.parse_args()

    if args.operation == "train-synthetic":
        train_synthetic_fixture(
            candidate_dir=Path("/candidate"),
            resolved_manifest_path=Path("/resolved_manifest.yaml"),
            outputs_dir=Path("/outputs"),
            artifact_run_dir=Path("/"),
            max_prediction_samples=args.max_prediction_samples,
            prediction_sample_policy=args.prediction_sample_policy,
        )
    elif args.operation == "train-gvccs":
        train_gvccs(
            candidate_dir=Path("/candidate"),
            resolved_manifest_path=Path("/resolved_manifest.yaml"),
            outputs_dir=Path("/outputs"),
            artifact_run_dir=Path("/"),
            data_root=Path("/data"),
            max_samples=args.max_samples,
            max_prediction_samples=args.max_prediction_samples,
            prediction_sample_policy=args.prediction_sample_policy,
        )
    elif args.operation == "train-research-problem":
        provider_config = ResearchProblemProviderConfig.model_validate_json(args.provider_config_json)
        train_research_problem_run(
            Path("/"),
            provider_config,
            max_samples=args.max_samples,
            max_prediction_samples=args.max_prediction_samples,
            prediction_sample_policy=args.prediction_sample_policy,
        )
    elif args.operation == "evaluate-run":
        evaluate_run(
            Path("/"),
            backend=args.backend,
            data_root=args.data_root,
            max_artifact_samples=args.max_artifact_samples,
        )


if __name__ == "__main__":
    main()
