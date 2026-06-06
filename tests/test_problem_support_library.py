from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml_autoresearch.problem_support.imaging import (
    deterministic_photometric_perturbation,
    horizontal_flip_image_mask,
    mask_image_to_tensor,
    rasterize_coco_polygons,
    rgb_image_to_tensor,
)
from ml_autoresearch.problem_support.segmentation import (
    binary_segmentation_metrics,
    derive_boundary_target_v1,
    derive_line_target_v1,
)
from ml_autoresearch.problem_support.frame_sequences import infer_timestamped_frame_sequences


def test_problem_support_rasterizes_coco_polygons_and_converts_images_to_tensors() -> None:
    mask = rasterize_coco_polygons(width=4, height=4, segmentations=((0, 0, 2, 0, 2, 2, 0, 2),))

    assert mask.mode == "L"
    assert mask.size == (4, 4)
    mask_tensor = mask_image_to_tensor(mask)
    assert mask_tensor.shape == (1, 4, 4)
    assert mask_tensor.max().item() == 1.0

    image = Image.new("RGB", (2, 1), (255, 128, 0))
    rgb_tensor = rgb_image_to_tensor(image)
    assert rgb_tensor.shape == (3, 1, 2)
    assert torch.allclose(rgb_tensor[:, 0, 0], torch.tensor([1.0, 128 / 255, 0.0]))


def test_problem_support_image_mask_transforms_are_reusable_and_deterministic() -> None:
    image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 11
    mask = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)

    flipped_image, flipped_mask = horizontal_flip_image_mask(image, mask)
    assert torch.equal(flipped_image, image.flip(dims=[2]))
    assert torch.equal(flipped_mask, mask.flip(dims=[2]))

    perturbed_once = deterministic_photometric_perturbation(image, contrast=1.05, brightness=0.01, noise_seed=123)
    perturbed_twice = deterministic_photometric_perturbation(image, contrast=1.05, brightness=0.01, noise_seed=123)
    assert torch.equal(perturbed_once, perturbed_twice)
    assert torch.all((perturbed_once >= 0.0) & (perturbed_once <= 1.0))


def test_problem_support_segmentation_metrics_and_auxiliary_targets_are_reusable() -> None:
    target = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
    prediction = torch.tensor([[[[0, 1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)

    metrics = binary_segmentation_metrics(prediction, target)
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["iou"] == pytest.approx(0.5)

    assert derive_line_target_v1(target).sum().item() == 9
    assert derive_boundary_target_v1(target).sum().item() == 9


@dataclass(frozen=True)
class GenericFrame:
    frame_id: int
    image_path: Path


def test_problem_support_infers_timestamped_frame_sequences_without_gvccs_types() -> None:
    samples = [
        GenericFrame(1, Path("cam_20260601000000.png")),
        GenericFrame(2, Path("cam_20260601000030.png")),
        GenericFrame(3, Path("cam_20260601000200.png")),
        GenericFrame(4, Path("no_timestamp.png")),
    ]

    sequences = infer_timestamped_frame_sequences(samples, filename_for_item=lambda sample: sample.image_path.name)

    assert [[sample.frame_id for sample in sequence] for sequence in sequences] == [[1, 2], [3]]
