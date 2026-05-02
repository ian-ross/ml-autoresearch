import json
from pathlib import Path

from ml_autoresearch.runs import RunStatus, run_candidate_with_synthetic_fixture
from ml_autoresearch.synthetic import SyntheticContrailDataset


def write_trainable_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: trainable_candidate
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
        "        self.net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1))\n"
        "    def forward(self, x):\n"
        "        return {'mask_logits': self.net(x)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    return candidate


def test_synthetic_fixture_dataset_is_deterministic():
    first_image, first_mask = SyntheticContrailDataset(2, seed=123)[1]
    second_image, second_mask = SyntheticContrailDataset(2, seed=123)[1]

    assert first_image.equal(second_image)
    assert first_mask.equal(second_mask)
    assert first_image.shape == (3, 128, 128)
    assert first_mask.shape == (1, 128, 128)


def test_run_candidate_with_synthetic_fixture_writes_result_artifacts(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs")

    assert run.status == RunStatus.COMPLETED
    run_dir = run.run_dir
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["training_failure_reason"] is None
    assert (run_dir / "logs" / "training.log").read_text()
    assert (run_dir / "metrics.jsonl").read_text()
    final = json.loads((run_dir / "final_metrics.json").read_text())
    assert set(final) >= {"val/dice", "val/iou", "val/precision", "val/recall", "val/loss"}
