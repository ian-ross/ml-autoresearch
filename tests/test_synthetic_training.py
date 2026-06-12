import json
from pathlib import Path

from PIL import Image
import torch
import yaml

from ml_autoresearch.runs import RunStatus
from ml_autoresearch.synthetic import SyntheticContrailDataset
from research_problem_helpers import run_candidate_with_synthetic_fixture
from ml_autoresearch.training import (
    _best_validation_metrics,
    _data_loader_for_sampling,
    _dataset_with_augmentation_policy,
    derive_boundary_target_v1,
)


def write_trainable_candidate(root: Path, *, max_epochs: int = 1, augmentation_policy: str | None = None) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    data_block = "" if augmentation_policy is None else f"data:\n  augmentation_policy: {augmentation_policy}\n"
    (candidate / "manifest.yaml").write_text(
        f"""
name: trainable_candidate
input_mode: single_frame_rgb
output_form: mask_logits
{data_block}training:
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


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return index


def loader_order(policy: str) -> list[int]:
    loader = _data_loader_for_sampling(IndexDataset(10), batch_size=3, sampling_policy=policy)
    return [int(item) for batch in loader for item in batch]


def test_sampling_policy_sequential_preserves_dataset_order():
    assert loader_order("sequential") == list(range(10))


def test_sampling_policy_deterministic_shuffle_changes_training_order_reproducibly():
    first = loader_order("deterministic_shuffle")
    second = loader_order("deterministic_shuffle")

    assert first == second
    assert first != list(range(10))
    assert sorted(first) == list(range(10))


def test_sampling_policy_validation_loader_stays_stable():
    # Validation loaders must use sequential policy regardless of the manifest's training Sampling Policy.
    assert loader_order("sequential") == list(range(10))


def test_augmentation_policy_presets_are_harness_applied_to_training_samples():
    image, mask = SyntheticContrailDataset(2, seed=123)[1]
    augmented = _dataset_with_augmentation_policy(SyntheticContrailDataset(2, seed=123), "light_combined")

    augmented_image, augmented_mask = augmented[1]

    assert not torch.equal(augmented_image, image)
    assert augmented_image.shape == image.shape
    assert augmented_mask.shape == mask.shape
    assert torch.equal(augmented_mask, torch.flip(mask, dims=[2]))


def test_augmentation_policy_none_leaves_validation_samples_stable():
    image, mask = SyntheticContrailDataset(2, seed=123)[1]
    unaugmented = _dataset_with_augmentation_policy(SyntheticContrailDataset(2, seed=123), "none")

    stable_image, stable_mask = unaugmented[1]

    assert torch.equal(stable_image, image)
    assert torch.equal(stable_mask, mask)


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
    assert metadata["research_problem"]["id"] == "ground_camera_contrail_detection"
    assert metadata["research_problem"]["version"] == "v0"
    assert metadata["research_problem"]["contract_version"] == "v0"
    assert metadata["research_problem"]["provider"]["target"] == "gvccs.research_problem:build_spec"
    assert metadata["training_failure_reason"] is None
    assert metadata["artifacts"]["prediction_samples"] == "outputs/prediction_samples/samples.json"
    assert metadata["artifacts"]["best_metrics"] == "outputs/best_metrics.json"
    assert (run_dir / "outputs" / "logs" / "training.log").read_text()
    assert (run_dir / "outputs" / "metrics.jsonl").read_text()
    final = json.loads((run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"val/dice", "val/iou", "val/precision", "val/recall", "val/loss"}
    assert final["artifacts"]["prediction_samples"] == "outputs/prediction_samples/samples.json"
    assert final["artifacts"]["best_metrics"] == "outputs/best_metrics.json"
    assert final["artifacts"]["best_epoch_model"] == "outputs/models/best_epoch_model.pt"
    best_model_path = run_dir / "outputs" / "models" / "best_epoch_model.pt"
    assert best_model_path.is_file()
    checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["selection_metric"] == "val/dice"
    assert "model_state_dict" in checkpoint
    best = json.loads((run_dir / "outputs" / "best_metrics.json").read_text())
    assert best["selection_metric"] == "val/dice"
    assert best["selection_mode"] == "max"
    assert best["epoch"] == final["epoch"] == 1
    assert best["selection_value"] == final["val/dice"]
    assert best["model_artifact"] == "outputs/models/best_epoch_model.pt"
    assert best["metrics"]["val/dice"] == final["val/dice"]

    samples_dir = run_dir / "outputs" / "prediction_samples"
    samples = json.loads((samples_dir / "samples.json").read_text())
    assert samples["status"] == "completed"
    assert samples["split"] == "val"
    assert 0 < samples["sample_count"] <= samples["max_sample_count"]
    assert len(samples["samples"]) == samples["sample_count"]

    first = samples["samples"][0]
    assert first["sample_id"] == "val/000000"
    assert first["split"] == "val"
    assert set(first) >= {"dice", "iou", "paths"}
    assert first["paths"] == {
        "input": "sample_000_input.png",
        "ground_truth": "sample_000_ground_truth.png",
        "prediction": "sample_000_prediction.png",
        "overlay": "sample_000_overlay.png",
        "probability_heatmap": "sample_000_probability_heatmap.png",
    }
    sizes = []
    for relative_path in first["paths"].values():
        png = samples_dir / relative_path
        assert png.is_file()
        with Image.open(png) as image:
            sizes.append(image.size)
    assert sizes == [(128, 128)] * 5


def test_synthetic_fixture_training_applies_selected_augmentation_policy(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path, augmentation_policy="light_combined")

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    resolved = yaml.safe_load((run.run_dir / "resolved_manifest.yaml").read_text())
    assert resolved["data"]["augmentation_policy"] == "light_combined"
    assert resolved["data"]["augmentation_policy_effective"] == "light_combined"
    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"val/dice", "val/loss"}


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
    best = json.loads((run.run_dir / "outputs" / "best_metrics.json").read_text())
    checkpoint = torch.load(run.run_dir / best["model_artifact"], map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == best["epoch"]
    assert checkpoint["selection_metric"] == best["selection_metric"] == "val/dice"
    assert checkpoint["selection_value"] == best["selection_value"]


def test_best_validation_metrics_selects_highest_dice_not_final_epoch():
    best = _best_validation_metrics(
        [
            {"split": "val", "epoch": 1, "val/dice": 0.2, "val/iou": 0.1, "val/loss": 0.9},
            {"split": "val", "epoch": 2, "val/dice": 0.8, "val/iou": 0.7, "val/loss": 0.4},
            {"split": "val", "epoch": 3, "val/dice": 0.5, "val/iou": 0.3, "val/loss": 0.6},
        ],
        model_artifact="outputs/models/best_epoch_model.pt",
    )

    assert best["selection_metric"] == "val/dice"
    assert best["selection_mode"] == "max"
    assert best["epoch"] == 2
    assert best["selection_value"] == 0.8
    assert best["model_artifact"] == "outputs/models/best_epoch_model.pt"
    assert best["metrics"] == {"epoch": 2, "val/dice": 0.8, "val/iou": 0.7, "val/loss": 0.4}


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


def write_line_aux_trainable_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: line_aux_trainable_candidate
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: 0.25
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
        "        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())\n"
        "        self.mask = nn.Conv2d(4, 1, 1)\n"
        "        self.line = nn.Conv2d(4, 1, 1)\n"
        "    def forward(self, x):\n"
        "        features = self.encoder(x)\n"
        "        return {'mask_logits': self.mask(features), 'line_logits': self.line(features)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    return candidate


def write_boundary_aux_trainable_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: boundary_aux_trainable_candidate
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: boundary
    output: boundary_logits
    loss: weighted_bce
    weight: 0.10
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
        "        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())\n"
        "        self.mask = nn.Conv2d(4, 1, 1)\n"
        "        self.boundary = nn.Conv2d(4, 1, 1)\n"
        "    def forward(self, x):\n"
        "        features = self.encoder(x)\n"
        "        return {'mask_logits': self.mask(features), 'boundary_logits': self.boundary(features)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    return candidate


def test_boundary_target_derivation_is_deterministic_mask_edge_band():
    mask = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    mask[:, :, 1:4, 1:4] = 1.0

    boundary = derive_boundary_target_v1(mask)

    expected = torch.tensor(
        [[[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]]],
        dtype=torch.float32,
    )
    assert torch.equal(boundary, expected)
    assert torch.equal(derive_boundary_target_v1(torch.zeros_like(mask)), torch.zeros_like(mask))


def test_line_auxiliary_training_records_primary_auxiliary_and_total_losses(tmp_path: Path):
    candidate = write_line_aux_trainable_candidate(tmp_path)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    records = [json.loads(line) for line in (run.run_dir / "outputs" / "metrics.jsonl").read_text().splitlines()]
    train_record = next(record for record in records if record["split"] == "train")
    assert set(train_record) >= {"loss", "mask_loss", "aux/line_loss"}
    val_record = next(record for record in records if record["split"] == "val")
    assert set(val_record) >= {"val/loss", "val/aux/line_loss", "val/total_loss"}
    assert val_record["val/total_loss"] >= val_record["val/loss"]

    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"train/loss", "train/mask_loss", "train/aux/line_loss", "val/aux/line_loss", "val/total_loss"}
    assert final["val/loss"] == val_record["val/loss"]
    assert final["val/total_loss"] == val_record["val/total_loss"]
    samples = json.loads((run.run_dir / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["status"] == "completed"


def test_boundary_auxiliary_training_records_boundary_loss(tmp_path: Path):
    candidate = write_boundary_aux_trainable_candidate(tmp_path)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    records = [json.loads(line) for line in (run.run_dir / "outputs" / "metrics.jsonl").read_text().splitlines()]
    train_record = next(record for record in records if record["split"] == "train")
    assert set(train_record) >= {"loss", "mask_loss", "aux/boundary_loss"}
    val_record = next(record for record in records if record["split"] == "val")
    assert set(val_record) >= {"val/loss", "val/aux/boundary_loss", "val/total_loss"}
    assert val_record["val/total_loss"] >= val_record["val/loss"]

    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"train/aux/boundary_loss", "val/aux/boundary_loss", "val/total_loss"}
