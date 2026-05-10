import json
import re
from pathlib import Path

import yaml

from ml_autoresearch.runs import RunFailureClassification, RunStatus, submit_candidate, validate_run_failure_classification


def write_valid_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: single_frame_unet_baseline
description: Tiny single-frame mask-only baseline for harness validation.
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
        "import torch\n"
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(3, 1, kernel_size=1)\n"
        "    def forward(self, x):\n"
        "        return self.conv(x)\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    return candidate


def test_submit_candidate_creates_accepted_run_directory(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    run = submit_candidate(candidate, runs_root)

    assert run.status == RunStatus.ACCEPTED
    assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{6}", run.run_id)
    run_dir = runs_root / run.run_id
    assert run.run_dir == run_dir
    assert (run_dir / "candidate" / "manifest.yaml").read_text() == (candidate / "manifest.yaml").read_text()
    assert (run_dir / "candidate" / "model.py").read_text() == (candidate / "model.py").read_text()
    assert (run_dir / "outputs" / "logs" / "validation.log").exists()
    assert set(child.name for child in run_dir.iterdir()) == {"candidate", "outputs", "resolved_manifest.yaml", "run_metadata.json"}

    resolved = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    assert resolved["name"] == "single_frame_unet_baseline"
    assert resolved["description"] == "Tiny single-frame mask-only baseline for harness validation."
    assert resolved["input_mode"] == "single_frame_rgb"
    assert resolved["output_form"] == "mask_logits"
    assert resolved["auxiliary_targets"] == []
    assert resolved["data"]["sampling_policy"] == "sequential"
    assert resolved["training"]["loss"] == "bce_dice"

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["run_id"] == run.run_id
    assert metadata["status"] == "accepted"
    assert metadata["candidate_source"]["path"] == str(candidate.resolve())
    assert metadata["rejection_reason"] is None
    assert metadata["created_at"]
    assert metadata["updated_at"]


def test_submit_candidate_creates_unique_run_ids(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    first = submit_candidate(candidate, runs_root)
    second = submit_candidate(candidate, runs_root)

    assert first.run_id != second.run_id
    assert first.run_dir.exists()
    assert second.run_dir.exists()


def test_submit_candidate_records_rejected_run(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "train.sh").write_text("echo nope\n")
    runs_root = tmp_path / "runs"

    run = submit_candidate(candidate, runs_root)

    assert run.status == RunStatus.REJECTED
    assert run.rejection_reason
    assert "forbidden" in run.rejection_reason
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "rejected"
    assert metadata["rejection_reason"] == run.rejection_reason
    assert (run.run_dir / "outputs" / "logs" / "validation.log").read_text()
    assert not (run.run_dir / "candidate" / "train.sh").exists()


def test_rejected_run_persists_contract_violation_failure_classification(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "train.sh").write_text("echo nope\n")
    runs_root = tmp_path / "runs"

    run = submit_candidate(candidate, runs_root)

    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert run.status == RunStatus.REJECTED
    assert metadata["failure_classification"] == "contract_violation"
    assert metadata["rejection_reason"] == run.rejection_reason
    assert "contract_violation" in metadata["reserved_failure_classifications"]


def test_invalid_run_failure_classification_is_rejected():
    assert validate_run_failure_classification("resource_failure") == RunFailureClassification.RESOURCE_FAILURE

    try:
        validate_run_failure_classification("not-a-classification")
    except ValueError as exc:
        assert "invalid run failure classification" in str(exc)
    else:
        raise AssertionError("invalid classification was accepted")


def add_valid_proposal(candidate: Path) -> None:
    (candidate / "PROPOSAL.md").write_text(
        """\
## Hypothesis
Fix candidate implementation defect without changing the research hypothesis.

## Comparison Target
Compare against the original proposal baseline.

## Expected Effect
The repaired candidate should run successfully.

## Implementation Sketch
Repair only candidate code defects.

## Contract Features Used
single_frame_rgb input, mask_logits output, bce_dice loss.

## Budget Requested
One synthetic-fixture run.

## Success Criteria
The Run reaches training.

## Fallback/Next Decision
Open a new Experiment Proposal for scientific changes.
"""
    )


def add_repair_lineage(candidate: Path, *, name: str, original_proposal_id: str = "proposal-original") -> None:
    manifest = yaml.safe_load((candidate / "manifest.yaml").read_text())
    manifest["name"] = name
    manifest["repair"] = {
        "original_proposal_id": original_proposal_id,
        "original_candidate_id": "candidate-original",
        "motivating_run_id": "run_20260501_120000_abcdef",
        "failure_classification": "candidate_bug",
        "preserves_original_hypothesis": True,
        "preserves_comparison_target": True,
    }
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


def test_repair_candidate_lineage_is_recorded_in_run_metadata_and_ledger(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    add_valid_proposal(candidate)
    add_repair_lineage(candidate, name="repair_candidate_1")
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = submit_candidate(candidate, runs_root, ledger_path=ledger, require_proposal=True)

    assert run.status == RunStatus.ACCEPTED
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["repair_lineage"] == {
        "original_proposal_id": "proposal-original",
        "original_candidate_id": "candidate-original",
        "motivating_run_id": "run_20260501_120000_abcdef",
        "failure_classification": "candidate_bug",
        "preserves_original_hypothesis": True,
        "preserves_comparison_target": True,
    }
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    candidate_created = next(event for event in events if event["event_type"] == "candidate_created")
    assert candidate_created["repair_lineage"] == metadata["repair_lineage"]


def test_autonomous_mode_rejects_more_than_two_repair_candidates_per_original_proposal(tmp_path: Path):
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"
    for index in range(2):
        (tmp_path / f"accepted_{index}").mkdir()
        candidate = write_valid_candidate(tmp_path / f"accepted_{index}")
        add_valid_proposal(candidate)
        add_repair_lineage(candidate, name=f"repair_candidate_{index}")
        assert submit_candidate(candidate, runs_root, ledger_path=ledger, require_proposal=True).status == RunStatus.ACCEPTED

    (tmp_path / "third").mkdir()
    third = write_valid_candidate(tmp_path / "third")
    add_valid_proposal(third)
    add_repair_lineage(third, name="repair_candidate_3")

    run = submit_candidate(third, runs_root, ledger_path=ledger, require_proposal=True)

    assert run.status == RunStatus.REJECTED
    assert "at most two Repair Candidates" in str(run.rejection_reason)
    assert not (run.run_dir / "candidate").exists()


def test_manual_mode_does_not_apply_autonomous_repair_limit(tmp_path: Path):
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"
    for index in range(2):
        (tmp_path / f"accepted_{index}").mkdir()
        candidate = write_valid_candidate(tmp_path / f"accepted_{index}")
        add_repair_lineage(candidate, name=f"repair_candidate_{index}")
        assert submit_candidate(candidate, runs_root, ledger_path=ledger).status == RunStatus.ACCEPTED

    (tmp_path / "third").mkdir()
    third = write_valid_candidate(tmp_path / "third")
    add_repair_lineage(third, name="repair_candidate_3")

    assert submit_candidate(third, runs_root, ledger_path=ledger).status == RunStatus.ACCEPTED


def test_submitted_candidate_source_copy_is_not_overwritten_by_later_source_edits(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    run = submit_candidate(candidate, runs_root)
    submitted_model = run.run_dir / "candidate" / "model.py"
    original_submitted_text = submitted_model.read_text()

    (candidate / "model.py").write_text("# repaired local source\n")

    assert submitted_model.read_text() == original_submitted_text
    assert submitted_model.read_text() != (candidate / "model.py").read_text()
