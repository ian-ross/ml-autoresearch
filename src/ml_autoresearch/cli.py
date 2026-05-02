"""Command-line interface for ML Autoresearch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ml_autoresearch.runs import RunStatus, submit_candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ml-autoresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser(
        "submit-candidate",
        help="validate a local Candidate Experiment and create a Run",
    )
    submit_parser.add_argument("--candidate", required=True, type=Path)
    submit_parser.add_argument("--runs-root", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "submit-candidate":
        run = submit_candidate(args.candidate, args.runs_root)
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir),
                    "status": run.status.value,
                    "rejection_reason": run.rejection_reason,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if run.status == RunStatus.REJECTED else 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
