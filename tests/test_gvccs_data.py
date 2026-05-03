import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from ml_autoresearch.gvccs import GVCCSDataError, GVCCSDataset, discover_gvccs_samples, deterministic_train_val_split
from ml_autoresearch.runs import RunStatus, run_candidate_with_gvccs_data


FIXTURE_ROOT = Path("tests/fixtures/gvccs_like")


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


def test_gvccs_fixture_layout_and_loader_returns_128_rgb_and_binary_mask():
    samples = discover_gvccs_samples(FIXTURE_ROOT, split="train")

    assert len(samples) == 4
    assert samples[0].image_path.name == "image_20260502000000.png"

    dataset = GVCCSDataset(samples)
    image, mask = dataset[0]

    assert image.shape == (3, 128, 128)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert mask.shape == (1, 128, 128)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert float(mask.sum()) > 0.0


def test_gvccs_discovery_fails_clearly_for_missing_or_malformed_root(tmp_path: Path):
    with pytest.raises(GVCCSDataError, match="data root does not exist"):
        discover_gvccs_samples(tmp_path / "missing", split="train")

    malformed = tmp_path / "gvccs"
    (malformed / "train" / "images").mkdir(parents=True)
    with pytest.raises(GVCCSDataError, match="missing annotations.json"):
        discover_gvccs_samples(malformed, split="train")


def test_gvccs_deterministic_train_val_split_and_max_samples():
    samples = discover_gvccs_samples(FIXTURE_ROOT, split="train", max_samples=3)

    first = deterministic_train_val_split(samples, val_fraction=1 / 3, seed=123)
    second = deterministic_train_val_split(samples, val_fraction=1 / 3, seed=123)

    assert [s.image_id for s in first.train] == [s.image_id for s in second.train]
    assert [s.image_id for s in first.val] == [s.image_id for s in second.val]
    assert len(first.train) == 2
    assert len(first.val) == 1


def test_run_candidate_with_gvccs_fixture_trains_one_epoch(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path)

    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", FIXTURE_ROOT, max_samples=4, max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"val/dice", "val/iou", "val/precision", "val/recall", "val/loss"}
    assert final["artifacts"]["prediction_samples"] == "outputs/prediction_samples/samples.json"
    samples = json.loads((run.run_dir / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["sample_count"] == 1
    assert samples["max_sample_count"] == 1
    assert samples["samples"][0]["source_image_path"].endswith(".png")
    for relative_path in samples["samples"][0]["paths"].values():
        with Image.open(run.run_dir / "outputs" / "prediction_samples" / relative_path) as image:
            assert image.size == (128, 128)
    assert "GVCCS" in (run.run_dir / "outputs" / "logs" / "training.log").read_text()
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["dataset"] == {
        "id": "gvccs",
        "host_data_path": str(FIXTURE_ROOT.resolve()),
        "container_data_path": "/data",
    }


def test_run_candidate_with_gvccs_data_validates_host_data_root_before_submission(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    with pytest.raises(GVCCSDataError, match="data root does not exist"):
        run_candidate_with_gvccs_data(candidate, runs_root, tmp_path / "missing")

    assert not runs_root.exists()


def test_malformed_gvccs_data_root_fails_run_with_training_metadata(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path)
    data_root = tmp_path / "malformed_gvccs"
    (data_root / "train" / "images").mkdir(parents=True)

    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", data_root)

    assert run.status == RunStatus.FAILED
    assert "missing annotations.json" in (run.rejection_reason or "")
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "missing annotations.json" in metadata["training_failure_reason"]
    assert metadata["dataset"] == {
        "id": "gvccs",
        "host_data_path": str(data_root.resolve()),
        "container_data_path": "/data",
    }
    assert "missing annotations.json" in (run.run_dir / "outputs" / "logs" / "training.log").read_text()
