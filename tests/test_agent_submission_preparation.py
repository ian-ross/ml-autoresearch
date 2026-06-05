import json
import subprocess
import sys
from pathlib import Path

from ml_autoresearch.agent_cli import app
from conftest import invoke_typer_cli


def write_candidate(root: Path, *, name: str = "agent_candidate", include_proposal: bool = True, include_readme: bool = True) -> Path:
    candidate = root / "drafts" / "candidates" / name
    candidate.mkdir(parents=True)
    (candidate / "manifest.yaml").write_text(
        f"""
name: {name}
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        "raise RuntimeError('model.py must not be imported during submission preparation')\n"
        "import torch\n"
    )
    if include_proposal:
        (candidate / "PROPOSAL.md").write_text(
            """\
## Hypothesis
Test static handoff.

## Comparison Target
Baseline candidate.

## Expected Effect
No execution during handoff.

## Implementation Sketch
Copy the validated draft.

## Contract Features Used
single_frame_rgb, mask_logits, bce_dice.

## Budget Requested
One later Harness validation.

## Success Criteria
Submission metadata is written.

## Fallback/Next Decision
Fix validation errors.
"""
        )
    if include_readme:
        (candidate / "README.md").write_text("# Agent candidate\n")
    return candidate


def run_agent_cli(*args: str):
    return invoke_typer_cli(app, args)


def test_prepare_candidate_submission_copies_draft_and_writes_metadata(tmp_path: Path):
    candidate = write_candidate(tmp_path)
    submissions_root = tmp_path / "submissions"

    completed = run_agent_cli("prepare-candidate-submission", "--candidate", str(candidate), "--submissions-root", str(submissions_root))

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "prepared"
    assert payload["candidate_id"] == "agent_candidate"
    submission_dir = submissions_root / "agent_candidate"
    copied_candidate = submission_dir / "candidate"
    assert (copied_candidate / "manifest.yaml").read_text() == (candidate / "manifest.yaml").read_text()
    assert (copied_candidate / "model.py").read_text() == (candidate / "model.py").read_text()
    assert (candidate / "manifest.yaml").is_file()
    metadata = json.loads((submission_dir / "submission.json").read_text())
    assert metadata == {
        "schema_version": "candidate_submission.v1",
        "submission_type": "candidate_experiment",
        "candidate_id": "agent_candidate",
        "candidate_path": "candidate",
        "requested_action": "validate_and_queue_for_harness_execution",
    }


def test_prepare_candidate_submission_refuses_to_overwrite_existing_submission(tmp_path: Path):
    candidate = write_candidate(tmp_path)
    existing = tmp_path / "submissions" / "agent_candidate"
    existing.mkdir(parents=True)

    completed = run_agent_cli("prepare-candidate-submission", "--candidate", str(candidate), "--submissions-root", str(tmp_path / "submissions"))

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "rejected"
    assert "already exists" in payload["rejection_reason"]


def test_prepare_candidate_submission_requires_proposal_and_readme(tmp_path: Path):
    missing_proposal = write_candidate(tmp_path / "missing-proposal", include_proposal=False)
    missing_readme = write_candidate(tmp_path / "missing-readme", include_readme=False)

    proposal_result = run_agent_cli(
        "prepare-candidate-submission", "--candidate", str(missing_proposal), "--submissions-root", str(tmp_path / "submissions-a")
    )
    readme_result = run_agent_cli(
        "prepare-candidate-submission", "--candidate", str(missing_readme), "--submissions-root", str(tmp_path / "submissions-b")
    )

    assert proposal_result.returncode == 1
    assert "PROPOSAL.md" in json.loads(proposal_result.stdout)["rejection_reason"]
    assert readme_result.returncode == 1
    assert "README.md" in json.loads(readme_result.stdout)["rejection_reason"]


def test_prepare_candidate_submission_rejects_invalid_candidate_and_name_mismatch(tmp_path: Path):
    invalid = write_candidate(tmp_path / "invalid")
    (invalid / "weights.pt").write_text("forbidden\n")
    mismatch = write_candidate(tmp_path / "mismatch", name="manifest_name")
    queue_id_path = mismatch.parent / "queue_id"
    mismatch.rename(queue_id_path)

    invalid_result = run_agent_cli(
        "prepare-candidate-submission", "--candidate", str(invalid), "--submissions-root", str(tmp_path / "submissions-a")
    )
    mismatch_result = run_agent_cli(
        "prepare-candidate-submission", "--candidate", str(queue_id_path), "--submissions-root", str(tmp_path / "submissions-b")
    )

    assert invalid_result.returncode == 1
    assert "forbidden" in json.loads(invalid_result.stdout)["rejection_reason"]
    assert mismatch_result.returncode == 1
    assert "manifest name" in json.loads(mismatch_result.stdout)["rejection_reason"]


def test_prepare_candidate_submission_is_static_without_torch_available(tmp_path: Path):
    candidate = write_candidate(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins, sys; "
                "real_import = builtins.__import__; "
                "exec(\"def guarded_import(name, *args, **kwargs):\\n"
                "     if name == 'torch' or name.startswith('torch.'):\\n"
                "         raise ModuleNotFoundError('No module named torch')\\n"
                "     return real_import(name, *args, **kwargs)\"); "
                "builtins.__import__ = guarded_import; "
                "from ml_autoresearch.agent_cli import main; "
                "main()"
            ),
            "prepare-candidate-submission",
            "--candidate",
            str(candidate),
            "--submissions-root",
            str(tmp_path / "submissions"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "prepared"
