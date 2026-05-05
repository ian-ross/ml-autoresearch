import json
from pathlib import Path

from PIL import Image
import torch

from ml_autoresearch.runs import RunStatus, run_candidate_with_synthetic_fixture
from ml_autoresearch.synthetic import SyntheticContrailDataset


def write_trainable_candidate(root: Path, *, max_epochs: int = 1) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        f"""
name: trainable_candidate
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
    assert metadata["artifacts"]["prediction_samples"] == "outputs/prediction_samples/samples.json"
    assert (run_dir / "outputs" / "logs" / "training.log").read_text()
    assert (run_dir / "outputs" / "metrics.jsonl").read_text()
    final = json.loads((run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"val/dice", "val/iou", "val/precision", "val/recall", "val/loss"}
    assert final["artifacts"]["prediction_samples"] == "outputs/prediction_samples/samples.json"

    samples_dir = run_dir / "outputs" / "prediction_samples"
    samples = json.loads((samples_dir / "samples.json").read_text())
    assert samples["status"] == "completed"
    assert samples["split"] == "val"
    assert samples["sample_count"] == 2
    assert len(samples["samples"]) == 2

    first = samples["samples"][0]
    assert first["sample_id"] == "val/000000"
    assert first["split"] == "val"
    assert set(first) >= {"dice", "iou", "paths"}
    assert first["paths"] == {
        "input": "sample_000_input.png",
        "ground_truth": "sample_000_ground_truth.png",
        "prediction": "sample_000_prediction.png",
        "overlay": "sample_000_overlay.png",
    }
    sizes = []
    for relative_path in first["paths"].values():
        png = samples_dir / relative_path
        assert png.is_file()
        with Image.open(png) as image:
            sizes.append(image.size)
    assert sizes == [(128, 128)] * 4


def test_synthetic_fixture_training_honors_manifest_max_epochs(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path, max_epochs=3)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    records = [json.loads(line) for line in (run.run_dir / "outputs" / "metrics.jsonl").read_text().splitlines()]
    train_epochs = [record["epoch"] for record in records if record["split"] == "train"]
    val_epochs = [record["epoch"] for record in records if record["split"] == "val"]
    assert sorted(set(train_epochs)) == [1, 2, 3]
    assert val_epochs == [1, 2, 3]

    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    final_val_record = [record for record in records if record["split"] == "val"][-1]
    assert final["epoch"] == 3
    assert final["val/loss"] == final_val_record["val/loss"]


def test_synthetic_fixture_training_uses_cuda_when_available(tmp_path: Path, monkeypatch):
    candidate = write_trainable_candidate(tmp_path, max_epochs=1)
    moved_modules: list[str] = []
    moved_tensors: list[str] = []

    original_module_to = torch.nn.Module.to
    original_tensor_to = torch.Tensor.to

    def record_module_to(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            moved_modules.append(str(args[0]))
            return self
        return original_module_to(self, *args, **kwargs)

    def record_tensor_to(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            moved_tensors.append(str(args[0]))
            return self
        return original_tensor_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.nn.Module, "to", record_module_to)
    monkeypatch.setattr(torch.Tensor, "to", record_tensor_to)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert final["hardware/device"] == "cuda"
    assert moved_modules
    assert moved_tensors
