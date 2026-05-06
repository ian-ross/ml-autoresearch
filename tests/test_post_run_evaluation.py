import json
import subprocess
import sys
from pathlib import Path

from ml_autoresearch.evaluations import evaluate_run
from ml_autoresearch.runs import RunStatus, run_candidate_with_gvccs_data


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


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_evaluate_run_api_writes_run_scoped_validation_artifacts(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED

    result = evaluate_run(run.run_dir, split="val", backend="native", data_root="tests/fixtures/gvccs_like")

    evaluation_dir = result.evaluation_dir
    assert evaluation_dir.parent == run.run_dir / "outputs" / "evaluations"
    assert evaluation_dir.name.startswith("eval_")
    metadata = json.loads((evaluation_dir / "evaluation_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["evaluation_id"] == evaluation_dir.name
    assert metadata["source_run"]["run_id"] == run.run_id
    assert metadata["split"] == "val"
    assert metadata["backend"] == "native"
    assert metadata["threshold"] == 0.5
    assert metadata["model_artifact"] == "outputs/models/best_epoch_model.pt"

    aggregate = json.loads((evaluation_dir / "aggregate_metrics.json").read_text())
    assert aggregate["split"] == "val"
    assert aggregate["threshold"] == 0.5
    assert aggregate["sample_count"] == 1
    assert set(aggregate["metrics"]) == {"dice", "iou", "precision", "recall"}

    records = [json.loads(line) for line in (evaluation_dir / "per_sample_metrics.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["sample_id"].startswith("val/")
    assert set(records[0]) >= {"dice", "iou", "precision", "recall", "image_path"}


def test_evaluate_run_api_records_failed_metadata_for_missing_model_artifact(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED
    (run.run_dir / "outputs" / "models" / "best_epoch_model.pt").unlink()

    try:
        evaluate_run(run.run_dir, split="val", backend="native", data_root="tests/fixtures/gvccs_like")
    except RuntimeError as exc:
        assert "model artifact is missing" in str(exc)
    else:
        raise AssertionError("expected missing model artifact failure")

    evaluation_dirs = sorted((run.run_dir / "outputs" / "evaluations").glob("eval_*"))
    assert len(evaluation_dirs) == 1
    metadata = json.loads((evaluation_dirs[0] / "evaluation_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "model artifact is missing" in metadata["failure_reason"]
    assert not (evaluation_dirs[0] / "aggregate_metrics.json").exists()


def test_evaluate_run_cli_uses_run_metadata_data_root_by_default(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    run = run_candidate_with_gvccs_data(candidate, runs_root, "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED

    completed = run_cli("evaluate-run", "--run", str(run.run_dir), "--split", "val", "--backend", "native")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["evaluation_id"].startswith("eval_")
    evaluation_dir = Path(payload["evaluation_dir"])
    assert (evaluation_dir / "evaluation_metadata.json").exists()
    assert (evaluation_dir / "aggregate_metrics.json").exists()
    assert (evaluation_dir / "per_sample_metrics.jsonl").exists()
