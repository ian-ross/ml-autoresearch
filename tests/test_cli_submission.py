import json
import subprocess
import sys
import time
from pathlib import Path


def write_valid_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: cli_candidate
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
    return candidate


def write_valid_candidate_with_proposal(root: Path) -> Path:
    candidate = write_valid_candidate(root)
    (candidate / "PROPOSAL.md").write_text(
        """\
## Hypothesis
This candidate should reduce false positives.

## Comparison Target
Compare against existing baseline run.

## Expected Effect
Improve val/dice.

## Implementation Sketch
Add one extra residual block.

## Contract Features Used
single_frame_rgb input, mask_logits output, bce_dice loss.

## Budget Requested
One run on synthetic fixture.

## Success Criteria
Increase val/dice by 0.01.

## Fallback/Next Decision
Keep baseline if no gain.
"""
    )
    return candidate


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_submit_candidate_cli_creates_run_and_prints_json(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "submit-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--no-require-proposal",
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "accepted"
    assert payload["run_id"].startswith("run_")
    assert (runs_root / payload["run_id"] / "run_metadata.json").exists()
    assert completed.stderr == ""


def test_submit_candidate_cli_defaults_to_require_proposal_in_autonomous_mode(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "submit-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "rejected"
    assert "PROPOSAL.md" in payload["rejection_reason"]


def test_submit_candidate_cli_exits_nonzero_for_rejected_candidate(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "weights.pt").write_text("nope\n")
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "submit-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--no-require-proposal",
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "rejected"
    assert "forbidden" in payload["rejection_reason"]
    assert (runs_root / payload["run_id"] / "run_metadata.json").exists()


def test_run_candidate_cli_rejects_missing_proposal_in_autonomous_mode(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "run-candidate", "--candidate", str(candidate), "--runs-root", str(runs_root), "--synthetic-fixture", "--backend", "native"
    )

    assert completed.returncode == 1, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "rejected"
    assert "PROPOSAL.md" in payload["rejection_reason"]


def test_run_candidate_cli_synthetic_fixture_trains_with_proposal_when_autonomous_and_accepts_by_default(tmp_path: Path):
    candidate = write_valid_candidate_with_proposal(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "run-candidate", "--candidate", str(candidate), "--runs-root", str(runs_root), "--synthetic-fixture", "--backend", "native"
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert (runs_root / payload["run_id"] / "outputs" / "final_metrics.json").exists()
    assert (runs_root / payload["run_id"] / "outputs" / "logs" / "training.log").exists()


def test_run_candidate_cli_accepts_max_prediction_samples(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--synthetic-fixture",
        "--max-prediction-samples",
        "3",
        "--backend",
        "native",
        "--no-require-proposal",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    samples = json.loads((runs_root / payload["run_id"] / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["max_sample_count"] == 3
    assert samples["sample_count"] == 3


def test_run_candidate_cli_can_daemonize_training(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--synthetic-fixture",
        "--backend",
        "native",
        "--daemonize",
        "--no-require-proposal",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "daemonized"
    log_path = Path(payload["log_path"])
    assert log_path.exists()

    deadline = time.time() + 20
    completed_runs = []
    while time.time() < deadline:
        completed_runs = [p for p in runs_root.glob("run_*/outputs/final_metrics.json")]
        if completed_runs:
            break
        time.sleep(0.25)
    assert completed_runs
    assert payload["pid"] > 0


def test_run_candidate_cli_gvccs_fixture_data_root_trains_and_prints_json(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--data-root",
        "tests/fixtures/gvccs_like",
        "--max-samples",
        "4",
        "--prediction-sample-policy",
        "adjacent_and_scattered",
        "--backend",
        "native",
        "--no-require-proposal",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert (runs_root / payload["run_id"] / "outputs" / "final_metrics.json").exists()
    samples = json.loads((runs_root / payload["run_id"] / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["prediction_sample_policy"] == "adjacent_and_scattered"
    assert "GVCCS" in (runs_root / payload["run_id"] / "outputs" / "logs" / "training.log").read_text()
