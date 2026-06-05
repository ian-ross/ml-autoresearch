"""Harness-owned GVCCS Dataset adapters for GVCCS Input Modes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from ml_autoresearch.errors import GVCCSDataError
from ml_autoresearch.problem_support.frame_sequences import infer_timestamped_frame_sequences, timestamp_from_filename
from ml_autoresearch.problem_support.imaging import mask_image_to_tensor, rasterize_coco_polygons, rgb_image_to_tensor

IMAGE_SIZE = 128
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SPLIT_SEED = 20260502


@dataclass(frozen=True)
class GVCCSSample:
    image_id: int
    image_path: Path
    width: int
    height: int
    segmentations: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class GVCCSSplit:
    train: list[GVCCSSample]
    val: list[GVCCSSample]


@dataclass(frozen=True)
class GVCCSTemporalClip:
    previous: GVCCSSample
    target: GVCCSSample
    next: GVCCSSample


class GVCCSDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """RGB image and binary Contrail Mask pairs resized to 128x128.

    Expected local GVCCS layout:

    data-root/
      train/
        images/image_*.jpg|png
        annotations.json   # COCO-style images/annotations with polygon segmentations
      test/                # optional for training
        images/
        annotations.json

    The repository fixture uses the same layout with generated PNG images.
    """

    def __init__(self, samples: list[GVCCSSample], *, image_size: int = IMAGE_SIZE) -> None:
        if not samples:
            raise GVCCSDataError("GVCCS dataset requires at least one discovered image/mask pair")
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _load_image_and_mask(self.samples[index], image_size=self.image_size)


class GVCCSTemporalClipDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """3-frame, stride-1 centered RGB clips and target-frame Contrail Masks.

    The clip layout is channel-stacked RGB: previous RGB, target RGB, next RGB,
    producing a 9HW tensor while the mask remains the target frame's 1HW mask.
    """

    def __init__(self, samples_or_clips: list[GVCCSSample] | list[GVCCSTemporalClip], *, image_size: int = IMAGE_SIZE) -> None:
        if not samples_or_clips:
            raise GVCCSDataError("GVCCS temporal clip dataset requires at least one sample or clip")
        first = samples_or_clips[0]
        if isinstance(first, GVCCSTemporalClip):
            self.clips = list(samples_or_clips)  # type: ignore[arg-type]
        else:
            self.clips = build_centered_temporal_clips(samples_or_clips)  # type: ignore[arg-type]
        if not self.clips:
            raise GVCCSDataError("GVCCS dataset does not contain any 3-frame stride-1 centered temporal clips")
        self.samples = [clip.target for clip in self.clips]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip = self.clips[index]
        previous, _ = _load_image_and_mask(clip.previous, image_size=self.image_size)
        target, mask = _load_image_and_mask(clip.target, image_size=self.image_size)
        next_image, _ = _load_image_and_mask(clip.next, image_size=self.image_size)
        return torch.cat([previous, target, next_image], dim=0), mask


def discover_gvccs_samples(data_root: str | Path, *, split: str = "train", max_samples: int | None = None) -> list[GVCCSSample]:
    """Discover image/Contrail Mask pairs from a local GVCCS Dataset root."""

    root = Path(data_root)
    if not root.exists():
        raise GVCCSDataError(f"GVCCS data root does not exist: {root}")
    if not root.is_dir():
        raise GVCCSDataError(f"GVCCS data root is not a directory: {root}")

    split_dir = root / split
    images_dir = split_dir / "images"
    annotations_path = split_dir / "annotations.json"
    if not split_dir.is_dir():
        raise GVCCSDataError(f"malformed GVCCS data root: missing {split}/ directory under {root}")
    if not images_dir.is_dir():
        raise GVCCSDataError(f"malformed GVCCS data root: missing {split}/images directory under {root}")
    if not annotations_path.is_file():
        raise GVCCSDataError(f"malformed GVCCS data root: missing annotations.json at {annotations_path}")

    payload = _read_annotations(annotations_path)
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise GVCCSDataError(f"malformed GVCCS annotations: expected images and annotations lists in {annotations_path}")

    segmentations_by_image_id = _segmentations_by_image_id(annotations, annotations_path)
    samples: list[GVCCSSample] = []
    for image_record in images:
        if not isinstance(image_record, dict):
            raise GVCCSDataError(f"malformed GVCCS annotations: image record is not an object in {annotations_path}")
        sample = _sample_from_image_record(image_record, images_dir, segmentations_by_image_id, annotations_path)
        samples.append(sample)

    samples.sort(key=lambda item: (item.image_id, item.image_path.name))
    if max_samples is not None:
        if max_samples < 1:
            raise GVCCSDataError("max_samples must be at least 1")
        samples = samples[:max_samples]
    if not samples:
        raise GVCCSDataError(f"no GVCCS image/mask pairs discovered in {split_dir}")
    return samples


def build_centered_temporal_clips(samples: list[GVCCSSample]) -> list[GVCCSTemporalClip]:
    """Build 3-frame, stride-1 centered clips within inferred GVCCS Frame Sequences."""

    clips: list[GVCCSTemporalClip] = []
    for sequence in infer_frame_sequences(samples):
        if len(sequence) < 3:
            continue
        for index in range(1, len(sequence) - 1):
            clips.append(GVCCSTemporalClip(sequence[index - 1], sequence[index], sequence[index + 1]))
    return clips


def infer_frame_sequences(samples: list[GVCCSSample]) -> list[list[GVCCSSample]]:
    """Infer GVCCS Frame Sequences from timestamp-like filenames.

    Samples are sorted by the first 14-digit timestamp in the filename. A new
    Frame Sequence starts when adjacent timestamps differ by more than 30 seconds.
    """

    return infer_timestamped_frame_sequences(samples, filename_for_item=lambda sample: sample.image_path.name)


_timestamp_from_filename = timestamp_from_filename


def deterministic_train_val_split(
    samples: list[GVCCSSample], *, val_fraction: float = 0.2, seed: int = DEFAULT_SPLIT_SEED
) -> GVCCSSplit:
    if not samples:
        raise GVCCSDataError("cannot split empty GVCCS sample list")
    if not 0.0 < val_fraction < 1.0:
        raise GVCCSDataError("val_fraction must be between 0 and 1")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, round(len(samples) * val_fraction)) if len(samples) > 1 else 1
    if val_count >= len(samples) and len(samples) > 1:
        val_count = len(samples) - 1
    val_indices = set(indices[:val_count])
    train = [sample for index, sample in enumerate(samples) if index not in val_indices]
    val = [sample for index, sample in enumerate(samples) if index in val_indices]
    if not train:
        train = list(val)
    return GVCCSSplit(train=train, val=val)


def _read_annotations(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise GVCCSDataError(f"malformed GVCCS annotations JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GVCCSDataError(f"malformed GVCCS annotations: root must be an object in {path}")
    return payload


def _segmentations_by_image_id(annotations: list[object], path: Path) -> dict[int, list[tuple[float, ...]]]:
    result: dict[int, list[tuple[float, ...]]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise GVCCSDataError(f"malformed GVCCS annotations: annotation record is not an object in {path}")
        image_id = annotation.get("image_id")
        if not isinstance(image_id, int):
            raise GVCCSDataError(f"malformed GVCCS annotations: annotation image_id must be an integer in {path}")
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            raise GVCCSDataError(f"malformed GVCCS annotations: segmentation must be a list in {path}")
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                raise GVCCSDataError(f"malformed GVCCS annotations: segmentation polygon must contain x/y pairs in {path}")
            result.setdefault(image_id, []).append(tuple(float(value) for value in polygon))
    return result


def _sample_from_image_record(
    image_record: dict[str, object],
    images_dir: Path,
    segmentations_by_image_id: dict[int, list[tuple[float, ...]]],
    annotations_path: Path,
) -> GVCCSSample:
    image_id = image_record.get("id")
    file_name = image_record.get("file_name")
    width = image_record.get("width")
    height = image_record.get("height")
    if not isinstance(image_id, int):
        raise GVCCSDataError(f"malformed GVCCS annotations: image id must be an integer in {annotations_path}")
    if not isinstance(file_name, str) or not file_name:
        raise GVCCSDataError(f"malformed GVCCS annotations: image file_name must be a string in {annotations_path}")
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise GVCCSDataError(f"malformed GVCCS annotations: image width/height must be positive integers in {annotations_path}")
    image_path = images_dir / file_name
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise GVCCSDataError(f"unsupported GVCCS image extension for {image_path}; expected jpg/jpeg/png")
    if not image_path.is_file():
        raise GVCCSDataError(f"malformed GVCCS data root: annotation references missing image file {image_path}")
    return GVCCSSample(
        image_id=image_id,
        image_path=image_path,
        width=width,
        height=height,
        segmentations=tuple(segmentations_by_image_id.get(image_id, [])),
    )


def _load_image_and_mask(sample: GVCCSSample, *, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(sample.image_path).convert("RGB")
    mask = _rasterize_mask(sample)
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if mask.size != (image_size, image_size):
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    return _rgb_image_to_tensor(image), _mask_image_to_tensor(mask)


def _rasterize_mask(sample: GVCCSSample) -> Image.Image:
    return rasterize_coco_polygons(width=sample.width, height=sample.height, segmentations=sample.segmentations)


_rgb_image_to_tensor = rgb_image_to_tensor
_mask_image_to_tensor = mask_image_to_tensor
