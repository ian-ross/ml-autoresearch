from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ml_autoresearch.research_problems import get_research_problem_spec
from ml_autoresearch.research_problem_packages.gvccs import discover_gvccs_samples, select_gvccs_frames


FIXTURE_ROOT = Path("tests/fixtures/gvccs_like")


def test_gvccs_research_problem_adapter_supplies_metadata_and_single_frame_datasets(tmp_path: Path) -> None:
    manifest = tmp_path / "resolved_manifest.yaml"
    manifest.write_text("input_mode: single_frame_rgb\ndata:\n  frame_selection_policy_effective: all_target_frames\n")
    adapter = get_research_problem_spec("ground_camera_contrail_detection").training_adapter

    assert adapter is not None
    assert adapter.dataset_metadata({"dataset_root": str(FIXTURE_ROOT)}) == {
        "id": "gvccs",
        "host_data_path": str(FIXTURE_ROOT.resolve()),
        "container_data_path": "/data",
    }
    datasets = adapter.build_datasets(data_config={"dataset_root": str(FIXTURE_ROOT)}, resolved_manifest_path=manifest, max_samples=4)

    assert len(datasets.train_dataset) == 3
    assert len(datasets.validation_dataset) == 1
    assert datasets.data_policy_metadata["frame_selection_policy_effective"] == "all_target_frames"


def test_gvccs_research_problem_adapter_preserves_temporal_eligible_centers(tmp_path: Path) -> None:
    manifest = tmp_path / "resolved_manifest.yaml"
    manifest.write_text("input_mode: centered_temporal_rgb_clip\ndata:\n  frame_selection_policy_effective: temporal_eligible_center\n")
    adapter = get_research_problem_spec("ground_camera_contrail_detection").training_adapter
    assert adapter is not None
    data_root = _write_temporal_fixture(tmp_path)
    samples = discover_gvccs_samples(data_root, split="train")

    datasets = adapter.build_datasets(data_config={"dataset_root": str(data_root)}, resolved_manifest_path=manifest)
    adapter_centers = [clip.target.image_path.name for clip in [*datasets.train_dataset.clips, *datasets.validation_dataset.clips]]
    expected_centers = [sample.image_path.name for sample in select_gvccs_frames(samples, "temporal_eligible_center")]

    assert sorted(adapter_centers) == sorted(expected_centers)


def _write_temporal_fixture(root: Path, *, frame_count: int = 5) -> Path:
    data_root = root / "gvccs_temporal"
    images_dir = data_root / "train" / "images"
    images_dir.mkdir(parents=True)
    images = []
    annotations = []
    for index in range(frame_count):
        minute, second = divmod(index * 30, 60)
        file_name = f"image_2026050200{minute:02d}{second:02d}.png"
        Image.new("RGB", (4, 4), (index, index, index)).save(images_dir / file_name)
        images.append({"id": index, "file_name": file_name, "width": 4, "height": 4})
        if index == 2:
            annotations.append({"id": 1, "image_id": index, "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]]})
    (data_root / "train" / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "contrail"}]})
    )
    return data_root


def test_reusable_training_and_evaluation_modules_do_not_import_gvccs_dataset_types() -> None:
    reusable_paths = [
        Path("src/ml_autoresearch/training.py"),
        Path("src/ml_autoresearch/evaluations.py"),
        Path("src/ml_autoresearch/artifacts.py"),
    ]
    forbidden = (
        "ml_autoresearch.gvccs",
        "GVCCSDataset",
        "GVCCSTemporalClipDataset",
        "discover_gvccs_samples",
        "infer_timestamped_frame_sequences",
    )

    for path in reusable_paths:
        source = path.read_text()
        assert not any(term in source for term in forbidden), path
