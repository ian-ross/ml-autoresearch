import json
import time
from pathlib import Path

import pytest

from ml_autoresearch.batches import (
    ExperimentBatchError,
    get_batch_summary,
    list_batches,
    run_experiment_batch_with_synthetic_fixture,
)
from ml_autoresearch.errors import TrainingError
from ml_autoresearch.execution import OperationResult


class FastBackend:
    name = "fast-test"

    def __init__(self, *, fail_candidate: str | None = None, sleep_seconds: float = 0.0):
        self.fail_candidate = fail_candidate
        self.sleep_seconds = sleep_seconds

    def smoke_test(self, run_dir):
        return OperationResult(backend=self.name, operation="smoke_test")

    def train_synthetic(self, run_dir, *, max_prediction_samples=2, prediction_sample_policy="first_n"):
        path = Path(run_dir)
        candidate_name = json.loads((path / "run_metadata.json").read_text())["candidate_source"]["path"].split("/")[-1]
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        if candidate_name == self.fail_candidate:
            raise TrainingError("intentional candidate failure")
        outputs = path / "outputs"
        (outputs / "logs").mkdir(parents=True, exist_ok=True)
        (outputs / "logs" / "training.log").write_text("trained\n")
        (outputs / "metrics.jsonl").write_text('{"val/dice": 0.5}\n')
        (outputs / "best_metrics.json").write_text(
            json.dumps({"selection_metric": "val/dice", "selection_value": 0.5, "metrics": {"val/dice": 0.5}}) + "\n"
        )
        (outputs / "final_metrics.json").write_text(json.dumps({"val/dice": 0.5}) + "\n")
        return OperationResult(backend=self.name, operation="train_synthetic")

    def train_gvccs(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate_run(self, *args, **kwargs):
        raise NotImplementedError


def write_candidate(root: Path, name: str) -> Path:
    candidate = root / name
    candidate.mkdir(parents=True)
    (candidate / "manifest.yaml").write_text(
        f"""
name: {name}
description: Batch variant {name}.
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
    (candidate / "README.md").write_text(f"# {name}\n")
    return candidate


def write_batch(root: Path, names: list[str]) -> Path:
    batch = root / "experiment_batch"
    candidates = batch / "candidates"
    candidates.mkdir(parents=True)
    (batch / "BATCH_PROPOSAL.md").write_text(
        """# Batch Proposal

## Shared hypothesis
Small variants should train.

## Shared comparison target
Baseline run.

## Decision criteria
Compare validation Dice.

## Success criteria
At least one variant completes.
"""
    )
    for name in names:
        write_candidate(candidates, name)
    return batch


def test_valid_experiment_batch_runs_all_candidates_and_writes_summary(tmp_path: Path):
    batch_dir = write_batch(tmp_path, ["variant_a", "variant_b"])

    result = run_experiment_batch_with_synthetic_fixture(
        batch_dir,
        batches_root=tmp_path / "batches",
        runs_root=tmp_path / "runs",
        backend=FastBackend(),
        ledger_path=tmp_path / "research-ledger.jsonl",
    )

    assert result["status"] == "completed"
    assert result["batch_id"].startswith("batch_")
    assert len(result["runs"]) == 2
    summary = get_batch_summary(tmp_path / "batches", result["batch_id"])
    assert summary["status"] == "completed"
    assert {item["candidate_id"] for item in summary["runs"]} == {"variant_a", "variant_b"}
    assert {item["status"] for item in summary["runs"]} == {"completed"}
    for item in summary["runs"]:
        metadata = json.loads(((tmp_path / "runs" / item["run_id"] / "run_metadata.json").read_text()))
        assert metadata["batch_id"] == result["batch_id"]
    assert [item["batch_id"] for item in list_batches(tmp_path / "batches")] == [result["batch_id"]]
    ledger_events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert "experiment_batch_created" in {event["event_type"] for event in ledger_events}
    assert "experiment_batch_completed" in {event["event_type"] for event in ledger_events}


def test_experiment_batch_rejects_oversized_batch_before_creating_runs(tmp_path: Path):
    batch_dir = write_batch(tmp_path, ["a", "b", "c", "d", "e"])

    with pytest.raises(ExperimentBatchError, match="at most 4"):
        run_experiment_batch_with_synthetic_fixture(
            batch_dir,
            batches_root=tmp_path / "batches",
            runs_root=tmp_path / "runs",
            backend=FastBackend(),
            ledger_path=tmp_path / "research-ledger.jsonl",
        )

    assert not (tmp_path / "runs").exists()


def test_experiment_batch_static_validation_is_all_or_nothing(tmp_path: Path):
    batch_dir = write_batch(tmp_path, ["valid", "invalid"])
    (batch_dir / "candidates" / "invalid" / "train.sh").write_text("echo nope\n")

    with pytest.raises(ExperimentBatchError, match="invalid"):
        run_experiment_batch_with_synthetic_fixture(
            batch_dir,
            batches_root=tmp_path / "batches",
            runs_root=tmp_path / "runs",
            backend=FastBackend(),
            ledger_path=tmp_path / "research-ledger.jsonl",
        )

    assert not (tmp_path / "runs").exists()


def test_experiment_batch_keeps_siblings_after_post_start_run_failure(tmp_path: Path):
    batch_dir = write_batch(tmp_path, ["variant_ok", "variant_fails"])

    result = run_experiment_batch_with_synthetic_fixture(
        batch_dir,
        batches_root=tmp_path / "batches",
        runs_root=tmp_path / "runs",
        backend=FastBackend(fail_candidate="variant_fails"),
        ledger_path=tmp_path / "research-ledger.jsonl",
    )

    assert result["status"] == "partially_failed"
    statuses = {item["candidate_id"]: item["status"] for item in result["runs"]}
    assert statuses == {"variant_ok": "completed", "variant_fails": "failed"}
