import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from gvccs import (
    GVCCSDataError,
    GVCCSDataset,
    GVCCSTemporalClipDataset,
    discover_gvccs_samples,
    deterministic_train_val_split,
    select_gvccs_frames,
)
from ml_autoresearch.runs import RunStatus, run_candidate_with_gvccs_data


FIXTURE_ROOT = Path("tests/fixtures/gvccs_like")


def _write_temporal_fixture(root: Path, *, frame_count: int = 5) -> Path:
    data_root = root / "gvccs_temporal"
    images_dir = data_root / "train" / "images"
    images_dir.mkdir(parents=True)
    images = []
    annotations = []
    for index in range(frame_count):
        minute, second = divmod(index * 30, 60)
        file_name = f"image_2026050200{minute:02d}{second:02d}.png"
        Image.new("RGB", (4, 4), (index * 30, index * 30 + 1, index * 30 + 2)).save(images_dir / file_name)
        images.append({"id": index, "file_name": file_name, "width": 4, "height": 4})
        if index == 2:
            annotations.append({"id": 1, "image_id": index, "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]]})
    (data_root / "train" / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "contrail"}]})
    )
    return data_root


def write_trainable_candidate(root: Path, *, max_epochs: int = 1, frame_selection_policy: str | None = None) -> Path:
    candidate = root / "candidate"
    candidate.mkdir(parents=True)
    data_block = "" if frame_selection_policy is None else f"data:\n  frame_selection_policy: {frame_selection_policy}\n"
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


def test_gvccs_temporal_clip_dataset_returns_centered_channel_stacked_clip(tmp_path: Path):
    samples = discover_gvccs_samples(_write_temporal_fixture(tmp_path), split="train")

    dataset = GVCCSTemporalClipDataset(samples, image_size=4)
    clip, mask = dataset[1]

    assert len(dataset) == 3
    assert clip.shape == (9, 4, 4)
    assert mask.shape == (1, 4, 4)
    assert dataset.samples[1].image_path.name == "image_20260502000100.png"
    assert clip[:, 0, 0].tolist() == pytest.approx(
        [30 / 255, 31 / 255, 32 / 255, 60 / 255, 61 / 255, 62 / 255, 90 / 255, 91 / 255, 92 / 255]
    )
    assert float(mask.sum()) > 0.0


def test_gvccs_temporal_clip_dataset_does_not_cross_frame_sequence_gaps(tmp_path: Path):
    data_root = _write_temporal_fixture(tmp_path, frame_count=3)
    annotations_path = data_root / "train" / "annotations.json"
    payload = json.loads(annotations_path.read_text())
    payload["images"][2]["file_name"] = "image_20260502000200.png"
    (data_root / "train" / "images" / "image_20260502000100.png").rename(data_root / "train" / "images" / "image_20260502000200.png")
    annotations_path.write_text(json.dumps(payload))
    samples = discover_gvccs_samples(data_root, split="train")

    with pytest.raises(GVCCSDataError, match="centered temporal clips"):
        GVCCSTemporalClipDataset(samples, image_size=4)


def test_temporal_eligible_frame_selection_excludes_sequence_boundaries_and_gaps(tmp_path: Path):
    data_root = _write_temporal_fixture(tmp_path, frame_count=6)
    annotations_path = data_root / "train" / "annotations.json"
    payload = json.loads(annotations_path.read_text())
    payload["images"][5]["file_name"] = "image_20260502000400.png"
    (data_root / "train" / "images" / "image_20260502000230.png").rename(data_root / "train" / "images" / "image_20260502000400.png")
    annotations_path.write_text(json.dumps(payload))
    samples = discover_gvccs_samples(data_root, split="train")

    selected = select_gvccs_frames(samples, "temporal_eligible_center")

    assert [sample.image_path.name for sample in selected] == [
        "image_20260502000030.png",
        "image_20260502000100.png",
        "image_20260502000130.png",
    ]


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


def test_run_temporal_candidate_with_gvccs_fixture_trains_and_writes_prediction_artifacts(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: temporal_trainable_candidate
input_mode: centered_temporal_rgb_clip
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
        "        self.net = nn.Sequential(nn.Conv2d(9, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 1, 1))\n"
        "    def forward(self, x):\n"
        "        return {'mask_logits': self.net(x)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    assert input_spec['mode'] == 'centered_temporal_rgb_clip'\n"
        "    assert input_spec['shape'] == [9, 128, 128]\n"
        "    assert input_spec['clip_length'] == 3\n"
        "    assert input_spec['layout'] == 'channel_stacked_rgb'\n"
        "    return Tiny()\n"
    )

    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", _write_temporal_fixture(tmp_path), max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["data_policy"]["frame_selection_policy_effective"] == "temporal_eligible_center"
    assert metadata["sample_counts"] == {"train": 2, "validation": 1}
    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"val/dice", "val/loss"}
    samples = json.loads((run.run_dir / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["sample_count"] == 1
    assert samples["samples"][0]["source_image_path"].endswith(".png")
    with Image.open(run.run_dir / "outputs" / "prediction_samples" / samples["samples"][0]["paths"]["input"]) as image:
        assert image.size == (128, 128)


def test_run_candidate_with_gvccs_fixture_trains_one_epoch(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path)

    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", FIXTURE_ROOT, max_samples=4, max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {"val/dice", "val/iou", "val/precision", "val/recall", "val/loss"}
    assert final["artifacts"]["prediction_samples"] == "outputs/prediction_samples/samples.json"
    samples = json.loads((run.run_dir / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["prediction_sample_policy"] == "first_n"
    assert samples["sample_count"] == 1
    assert samples["max_sample_count"] == 1
    assert samples["samples"][0]["source_image_path"].endswith(".png")
    for relative_path in samples["samples"][0]["paths"].values():
        with Image.open(run.run_dir / "outputs" / "prediction_samples" / relative_path) as image:
            assert image.size == (128, 128)
    assert "GVCCS" in (run.run_dir / "outputs" / "logs" / "training.log").read_text()
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["research_problem"]["id"] == "ground_camera_contrail_detection"
    assert metadata["research_problem"]["version"] == "v0"
    assert metadata["research_problem"]["contract_version"] == "v0"
    assert metadata["research_problem"]["provider"]["resolved_package_root"] == "/home/iross/code/gvccs-research-problem"
    assert metadata["dataset"] == {
        "id": "gvccs",
        "host_data_path": str(FIXTURE_ROOT.resolve()),
        "container_data_path": "/data",
    }
    assert metadata["data_policy"]["frame_selection_policy_effective"] == "all_target_frames"
    assert metadata["sample_counts"] == {"train": 3, "validation": 1}


def test_matched_single_frame_control_uses_temporal_eligible_target_subset(tmp_path: Path):
    data_root = _write_temporal_fixture(tmp_path, frame_count=5)
    samples = discover_gvccs_samples(data_root, split="train")
    assert [sample.image_path.name for sample in select_gvccs_frames(samples, "temporal_eligible_center")] == [
        clip.target.image_path.name for clip in GVCCSTemporalClipDataset(samples).clips
    ]
    single = write_trainable_candidate(tmp_path / "single", frame_selection_policy="temporal_eligible_center")

    run = run_candidate_with_gvccs_data(single, tmp_path / "runs", data_root, max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["data_policy"]["frame_selection_policy_effective"] == "temporal_eligible_center"
    assert metadata["sample_counts"] == {"train": 2, "validation": 1}
    prediction_samples = json.loads((run.run_dir / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert prediction_samples["samples"][0]["source_image_path"].endswith("image_20260502000030.png")


def test_run_candidate_with_gvccs_fixture_accepts_adjacent_and_scattered_prediction_sample_policy(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path)

    run = run_candidate_with_gvccs_data(
        candidate,
        tmp_path / "runs",
        FIXTURE_ROOT,
        max_samples=4,
        max_prediction_samples=1,
        prediction_sample_policy="adjacent_and_scattered",
    )

    assert run.status == RunStatus.COMPLETED
    samples = json.loads((run.run_dir / "outputs" / "prediction_samples" / "samples.json").read_text())
    assert samples["prediction_sample_policy"] == "adjacent_and_scattered"
    assert samples["sample_count"] == 1
    assert {sample["selection"]["selection_kind"] for sample in samples["samples"]} <= {
        "adjacent_window",
        "scattered_singleton",
    }


def test_run_candidate_with_gvccs_fixture_honors_manifest_max_epochs(tmp_path: Path):
    candidate = write_trainable_candidate(tmp_path, max_epochs=2)

    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", FIXTURE_ROOT, max_samples=4, max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    records = [json.loads(line) for line in (run.run_dir / "outputs" / "metrics.jsonl").read_text().splitlines()]
    assert sorted({record["epoch"] for record in records if record["split"] == "train"}) == [1, 2]
    assert [record["epoch"] for record in records if record["split"] == "val"] == [1, 2]
    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert final["epoch"] == 2


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
