"""In-container operation runner for Docker-backed Candidate Execution Boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_autoresearch.training import train_gvccs, train_synthetic_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed ML Autoresearch container operations.")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    train_synthetic = subparsers.add_parser("train-synthetic")
    train_synthetic.add_argument("--max-prediction-samples", type=int, default=2)
    train_gvccs_parser = subparsers.add_parser("train-gvccs")
    train_gvccs_parser.add_argument("--max-samples", type=int, default=None)
    train_gvccs_parser.add_argument("--max-prediction-samples", type=int, default=2)
    args = parser.parse_args()

    if args.operation == "train-synthetic":
        train_synthetic_fixture(
            candidate_dir=Path("/candidate"),
            resolved_manifest_path=Path("/resolved_manifest.yaml"),
            outputs_dir=Path("/outputs"),
            artifact_run_dir=Path("/"),
            max_prediction_samples=args.max_prediction_samples,
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
        )


if __name__ == "__main__":
    main()
