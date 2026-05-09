"""Tests that the Run lifecycle emits Research Ledger events through the
same validation path as the `record-research-event` CLI/API."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ml_autoresearch.research_ledger import CANONICAL_RESEARCH_LEDGER
from ml_autoresearch.runs import (
    RunStatus,
    run_candidate_with_synthetic_fixture,
    submit_candidate,
)


def write_trainable_candidate(root: Path, *, max_epochs: int = 1, with_proposal: bool = False) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        f"""
name: ledger_lifecycle_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: {max_epochs}
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1))\n"
        "    def forward(self, x):\n"
        "        return {'mask_logits': self.net(x)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    if with_proposal:
        (candidate / "PROPOSAL.md").write_text(
            """\
## Hypothesis
Test model improves recall.

## Comparison Target
Compare against previous best baseline.

## Expected Effect
Increase Dice on the validation set.

## Implementation Sketch
Keep current architecture and only add extra block.

## Contract Features Used
Single-frame input, mask_logits output, synthetic dataset.

## Budget Requested
1 synthetic training hour.

## Success Criteria
Higher val/dice than control.

## Fallback/Next Decision
Try alternate seed if run fails.
"""
        )
    return candidate


def write_invalid_candidate(root: Path) -> Path:
    candidate = root / "invalid_candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: bad_candidate
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
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(3, 1, kernel_size=1)\n"
        "    def forward(self, x):\n"
        "        return {'mask_logits': self.conv(x)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    # Forbidden file triggers Candidate Experiment Contract rejection.
    (candidate / "train.sh").write_text("echo nope\n")
    return candidate


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_submit_candidate_records_candidate_created_and_submitted_events(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = submit_candidate(candidate, runs_root, ledger_path=ledger)

    assert run.status == RunStatus.ACCEPTED
    rows = read_jsonl(ledger)
    assert len(rows) == 2
    candidate_created_event, submitted_event = rows
    assert candidate_created_event["event_type"] == "candidate_created"
    assert candidate_created_event["candidate_id"] == "ledger_lifecycle_candidate"
    assert candidate_created_event["candidate_path"].endswith("candidate")
    assert candidate_created_event.get("proposal_id") is None
    assert submitted_event["event_type"] == "candidate_submitted"
    assert submitted_event["candidate_id"] == "ledger_lifecycle_candidate"
    assert submitted_event["run_id"] == run.run_id
    assert submitted_event["created_at"].endswith("Z")


def test_submit_candidate_records_proposal_created_and_candidate_events_when_proposal_is_present(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path, with_proposal=True)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = submit_candidate(candidate, runs_root, ledger_path=ledger)

    assert run.status == RunStatus.ACCEPTED
    rows = read_jsonl(ledger)
    assert len(rows) == 3
    proposal_event, candidate_event, submitted_event = rows
    assert proposal_event["event_type"] == "proposal_created"
    assert proposal_event["proposal_id"] == "ledger_lifecycle_candidate"
    assert proposal_event["candidate_id"] == "ledger_lifecycle_candidate"
    assert proposal_event["proposal_path"].endswith("PROPOSAL.md")
    assert candidate_event["event_type"] == "candidate_created"
    assert candidate_event["candidate_id"] == "ledger_lifecycle_candidate"
    assert candidate_event["candidate_path"].endswith("candidate")
    assert candidate_event["proposal_id"] == "ledger_lifecycle_candidate"
    assert submitted_event["event_type"] == "candidate_submitted"
    assert submitted_event["run_id"] == run.run_id

    assert (run.run_dir / "candidate" / "PROPOSAL.md").is_file()


def test_submit_candidate_rejects_missing_proposal_when_required(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = submit_candidate(candidate, runs_root, ledger_path=ledger, require_proposal=True)

    assert run.status == RunStatus.REJECTED
    assert not ledger.exists()


def test_submit_candidate_does_not_record_event_when_rejected(tmp_path: Path) -> None:
    candidate = write_invalid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = submit_candidate(candidate, runs_root, ledger_path=ledger)

    assert run.status == RunStatus.REJECTED
    assert not ledger.exists()


def test_submit_candidate_default_ledger_path_is_runs_root_sibling(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    run = submit_candidate(candidate, runs_root)

    assert run.status == RunStatus.ACCEPTED
    default_ledger = tmp_path / CANONICAL_RESEARCH_LEDGER
    rows = read_jsonl(default_ledger)
    assert len(rows) == 2
    assert rows[0]["event_type"] == "candidate_created"
    assert rows[1]["event_type"] == "candidate_submitted"
    assert rows[1]["run_id"] == run.run_id


def test_run_candidate_with_synthetic_fixture_emits_full_lifecycle_events(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = run_candidate_with_synthetic_fixture(candidate, runs_root, ledger_path=ledger)

    assert run.status == RunStatus.COMPLETED
    rows = read_jsonl(ledger)
    types = [event["event_type"] for event in rows]
    assert types == ["candidate_created", "candidate_submitted", "run_started", "run_completed"]

    candidate_created_event, submitted, started, completed = rows
    assert candidate_created_event["candidate_id"] == "ledger_lifecycle_candidate"
    assert submitted["candidate_id"] == "ledger_lifecycle_candidate"
    assert submitted["run_id"] == run.run_id
    assert started["run_id"] == run.run_id
    assert started["candidate_id"] == "ledger_lifecycle_candidate"
    assert completed["run_id"] == run.run_id
    metrics_path = Path(completed["metrics_path"])
    assert metrics_path.is_file()
    assert metrics_path == run.run_dir / "outputs" / "final_metrics.json"


def test_run_candidate_with_synthetic_fixture_records_proposal_and_candidate_events_when_required(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path, with_proposal=True)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = run_candidate_with_synthetic_fixture(
        candidate,
        runs_root,
        ledger_path=ledger,
        require_proposal=True,
    )

    assert run.status == RunStatus.COMPLETED
    rows = read_jsonl(ledger)
    types = [event["event_type"] for event in rows]
    assert types == ["proposal_created", "candidate_created", "candidate_submitted", "run_started", "run_completed"]
    assert rows[0]["candidate_id"] == "ledger_lifecycle_candidate"
    assert rows[0]["proposal_path"].endswith("PROPOSAL.md")
    assert rows[1]["proposal_id"] == "ledger_lifecycle_candidate"
    assert rows[1]["candidate_path"].endswith("candidate")
    assert (run.run_dir / "candidate" / "PROPOSAL.md").is_file()


def test_run_candidate_with_synthetic_fixture_rejects_missing_proposal_when_required(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    run = run_candidate_with_synthetic_fixture(
        candidate,
        runs_root,
        ledger_path=ledger,
        require_proposal=True,
    )

    assert run.status == RunStatus.REJECTED
    assert not ledger.exists()
    assert "PROPOSAL.md" in (run.rejection_reason or "")


def test_run_candidate_with_synthetic_fixture_records_run_failed_on_training_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    from ml_autoresearch import execution
    from ml_autoresearch.training import TrainingError

    def boom(self, run_dir, *, max_prediction_samples=2, prediction_sample_policy="first_n"):
        raise TrainingError("synthetic training exploded")

    monkeypatch.setattr(execution.NativeBackend, "train_synthetic", boom)

    run = run_candidate_with_synthetic_fixture(candidate, runs_root, ledger_path=ledger)

    assert run.status == RunStatus.FAILED
    rows = read_jsonl(ledger)
    types = [event["event_type"] for event in rows]
    assert types == ["candidate_created", "candidate_submitted", "run_started", "run_failed"]
    failed = rows[-1]
    assert failed["run_id"] == run.run_id
    assert "synthetic training exploded" in failed["error"]


def test_submit_candidate_cli_supports_ledger_path_option(tmp_path: Path) -> None:
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "research-ledger.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "submit-candidate",
            "--candidate",
            str(candidate),
            "--runs-root",
            str(runs_root),
            "--ledger-path",
            str(ledger),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "accepted"
    rows = read_jsonl(ledger)
    assert len(rows) == 2
    assert rows[0]["event_type"] == "candidate_created"
    assert rows[0]["candidate_id"] == "ledger_lifecycle_candidate"
    assert rows[1]["event_type"] == "candidate_submitted"
    assert rows[1]["run_id"] == payload["run_id"]
