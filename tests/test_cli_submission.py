import json
import subprocess
import sys
from pathlib import Path

from ml_autoresearch.cli import app
from ml_autoresearch.execution import OperationResult
from conftest import invoke_typer_cli
from research_problem_helpers import write_fake_candidate_execution_config, write_fake_research_problem_package


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



def write_fake_execution_config(root: Path) -> None:
    write_fake_research_problem_package(root)
    write_fake_candidate_execution_config(root)


def run_cli(*args: str):
    return invoke_typer_cli(app, args)


def run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_validate_candidate_cli_reports_static_success_and_requires_artifacts_when_requested(tmp_path: Path):
    candidate = write_valid_candidate_with_proposal(tmp_path)
    (candidate / "README.md").write_text("# CLI candidate\n")

    completed = run_cli(
        "validate-candidate",
        "--candidate",
        str(candidate),
        "--require-proposal",
        "--require-readme",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "valid"
    assert payload["manifest"]["name"] == "cli_candidate"


def test_validate_candidate_cli_reports_static_failure_without_traceback(tmp_path: Path):
    candidate = write_valid_candidate_with_proposal(tmp_path)

    completed = run_cli("validate-candidate", "--candidate", str(candidate), "--require-readme")

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "invalid"
    assert "README.md" in payload["reason"]
    assert "Traceback" not in completed.stderr


def test_validate_candidate_cli_defaults_to_require_proposal_in_normal_cli(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)

    completed = run_cli(
        "validate-candidate",
        "--candidate",
        str(candidate),
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "invalid"
    assert "PROPOSAL.md" in payload["reason"]


def test_submit_candidate_cli_creates_run_and_prints_json(tmp_path: Path, monkeypatch):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(
        "ml_autoresearch.execution.NativeBackend.smoke_test",
        lambda self, run_dir: OperationResult(backend="native", operation="smoke_test"),
    )

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


def test_submit_candidate_cli_reports_ledger_failure_without_traceback(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "submit-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--ledger-path",
        str(tmp_path),
        "--no-require-proposal",
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr
    assert str(tmp_path) in completed.stderr


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
    write_fake_execution_config(tmp_path)

    completed = run_cli(
        "run-candidate", "--candidate", str(candidate), "--runs-root", str(runs_root), "--project-root", str(tmp_path), "--backend", "native"
    )

    assert completed.returncode == 1, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "rejected"
    assert "PROPOSAL.md" in payload["rejection_reason"]


def test_run_candidate_cli_reports_ledger_failure_without_traceback(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    write_fake_execution_config(tmp_path)

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--project-root",
        str(tmp_path),
        "--backend",
        "native",
        "--ledger-path",
        str(tmp_path),
        "--no-require-proposal",
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr
    assert str(tmp_path) in completed.stderr


def test_run_candidate_cli_accepts_by_default_when_proposal_present(tmp_path: Path):
    candidate = write_valid_candidate_with_proposal(tmp_path)
    runs_root = tmp_path / "runs"
    write_fake_execution_config(tmp_path)

    completed = run_cli(
        "run-candidate", "--candidate", str(candidate), "--runs-root", str(runs_root), "--project-root", str(tmp_path), "--backend", "native"
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert (runs_root / payload["run_id"] / "outputs" / "final_metrics.json").exists()
    assert (runs_root / payload["run_id"] / "outputs" / "logs" / "training.log").exists()


def test_run_candidate_cli_accepts_max_prediction_samples(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    write_fake_execution_config(tmp_path)

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--project-root",
        str(tmp_path),
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
    assert 0 < samples["sample_count"] <= 3


def test_run_candidate_cli_can_daemonize_training(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    write_fake_execution_config(tmp_path)

    completed = run_cli_subprocess(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--project-root",
        str(tmp_path),
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

    # Completion of native synthetic training is covered by the foreground run-candidate tests.
    # The daemonization contract only needs to prove that the CLI detaches a child command,
    # omits --daemonize from that child, and prepares a log file for it.
    assert payload["pid"] > 0
    assert "--daemonize" not in payload["command"]


def test_run_candidate_cli_requires_provider_config(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--project-root",
        str(tmp_path),
        "--backend",
        "native",
        "--no-require-proposal",
    )

    assert completed.returncode == 1
    assert "candidate-execution.toml" in completed.stderr


def test_run_candidate_cli_uses_fallback_dataset_config_from_file(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    write_fake_research_problem_package(tmp_path)
    write_fake_candidate_execution_config(
        tmp_path,
        data_root=tmp_path / "datasets" / "synthetic",
        data_config={"fixture": "tiny", "sample_count": 4},
        sample_count=4,
    )
    (tmp_path / "datasets" / "synthetic").mkdir(parents=True, exist_ok=True)

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(candidate),
        "--runs-root",
        str(runs_root),
        "--project-root",
        str(tmp_path),
        "--max-samples",
        "4",
        "--backend",
        "native",
        "--max-prediction-samples",
        "2",
        "--no-require-proposal",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert (runs_root / payload["run_id"] / "outputs" / "final_metrics.json").exists()
    samples = json.loads((runs_root / payload["run_id"] / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["prediction_sample_policy"] == "first_n"
    assert "Starting fake Research Problem training." in (runs_root / payload["run_id"] / "outputs" / "logs" / "training.log").read_text()
