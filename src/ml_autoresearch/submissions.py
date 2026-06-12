"""Agent Control Boundary Candidate Submission Preparation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.research_problems import ResearchProblemSpecRegistry

SUBMISSION_SCHEMA_VERSION = "candidate_submission.v1"
SUBMISSION_TYPE = "candidate_experiment"
REQUESTED_ACTION = "validate_and_queue_for_harness_execution"
RELATIVE_CANDIDATE_PATH = "candidate"

BATCH_SUBMISSION_SCHEMA_VERSION = "experiment_batch_submission.v1"
BATCH_SUBMISSION_TYPE = "experiment_batch"
BATCH_REQUESTED_ACTION = "validate_and_queue_batch_for_harness_execution"
RELATIVE_BATCH_PATH = "experiment_batch"


class CandidateSubmissionPreparationError(ValueError):
    """Raised when a draft Candidate Experiment cannot be prepared for handoff."""


def prepare_experiment_batch_submission(batch_dir: str | Path, submissions_root: str | Path) -> dict[str, object]:
    """Validate and copy a draft Experiment Batch into the submission queue."""

    from ml_autoresearch.batches import ExperimentBatchError, validate_experiment_batch_directory

    batch_path = Path(batch_dir)
    batch_submission_id = batch_path.name
    if not batch_submission_id:
        raise CandidateSubmissionPreparationError("batch path must have a directory name")

    submission_dir = Path(submissions_root) / batch_submission_id
    if submission_dir.exists():
        raise CandidateSubmissionPreparationError(f"submission directory already exists: {submission_dir}")

    try:
        candidates = validate_experiment_batch_directory(batch_path)
    except ExperimentBatchError as exc:
        raise CandidateSubmissionPreparationError(str(exc)) from exc

    submission_dir.parent.mkdir(parents=True, exist_ok=True)
    copied_batch_path = submission_dir / RELATIVE_BATCH_PATH
    shutil.copytree(batch_path, copied_batch_path)

    metadata = {
        "schema_version": BATCH_SUBMISSION_SCHEMA_VERSION,
        "submission_type": BATCH_SUBMISSION_TYPE,
        "batch_submission_id": batch_submission_id,
        "batch_path": RELATIVE_BATCH_PATH,
        "requested_action": BATCH_REQUESTED_ACTION,
    }
    (submission_dir / "submission.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return {
        "status": "prepared",
        "batch_submission_id": batch_submission_id,
        "submission_dir": str(submission_dir),
        "batch_path": str(copied_batch_path),
        "metadata_path": str(submission_dir / "submission.json"),
        "candidate_count": len(candidates),
        "submission": metadata,
    }


def prepare_candidate_submission(
    candidate_dir: str | Path,
    submissions_root: str | Path,
    *,
    research_problem_registry: ResearchProblemSpecRegistry | None = None,
) -> dict[str, object]:
    """Validate and copy a draft Candidate Experiment into the submission queue.

    This is a static packaging operation for the Agent Control Boundary. It
    validates candidate source files and documentation, but does not import model
    code, smoke-test, train, evaluate, or invoke Docker.
    """

    candidate_path = Path(candidate_dir)
    candidate_id = candidate_path.name
    if not candidate_id:
        raise CandidateSubmissionPreparationError("candidate path must have a directory name")

    submission_dir = Path(submissions_root) / candidate_id
    if submission_dir.exists():
        raise CandidateSubmissionPreparationError(f"submission directory already exists: {submission_dir}")

    try:
        manifest = validate_candidate_directory(
            candidate_path,
            require_proposal=True,
            require_readme=True,
            research_problem_registry=research_problem_registry,
        )
    except CandidateValidationError as exc:
        raise CandidateSubmissionPreparationError(str(exc)) from exc

    if manifest.name != candidate_id:
        raise CandidateSubmissionPreparationError(
            f"manifest name must match queue candidate id '{candidate_id}' (got '{manifest.name}')"
        )

    submission_dir.parent.mkdir(parents=True, exist_ok=True)
    copied_candidate_path = submission_dir / RELATIVE_CANDIDATE_PATH
    shutil.copytree(candidate_path, copied_candidate_path)

    metadata = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "submission_type": SUBMISSION_TYPE,
        "candidate_id": candidate_id,
        "candidate_path": RELATIVE_CANDIDATE_PATH,
        "requested_action": REQUESTED_ACTION,
    }
    (submission_dir / "submission.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return {
        "status": "prepared",
        "candidate_id": candidate_id,
        "submission_dir": str(submission_dir),
        "candidate_path": str(copied_candidate_path),
        "metadata_path": str(submission_dir / "submission.json"),
        "submission": metadata,
    }
