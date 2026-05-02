"""Deterministic synthetic contrail-like segmentation fixture data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticContrailConfig:
    image_size: int = 128
    base_seed: int = 13_371
    noise_std: float = 0.025


class SyntheticContrailDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generated sky RGB images with thin bright line/curve Contrail Masks."""

    def __init__(self, length: int, *, seed: int, config: SyntheticContrailConfig | None = None) -> None:
        self.length = length
        self.seed = seed
        self.config = config or SyntheticContrailConfig()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        size = self.config.image_size
        image = _sky_background(size, generator)
        mask = torch.zeros((size, size), dtype=torch.float32)

        # Deterministically include some cloud-ish negatives, while keeping most
        # samples positive with 1-3 contrails.
        contrail_count = int(torch.randint(1, 4, (1,), generator=generator).item())
        if index % 7 == 0:
            contrail_count = 0
        for _ in range(contrail_count):
            line_mask = _random_line_or_curve_mask(size, generator)
            mask = torch.maximum(mask, line_mask)

        cloud = _cloudish_negative(size, generator)
        image = torch.clamp(image + 0.12 * cloud.unsqueeze(0), 0.0, 1.0)
        if mask.any():
            image = torch.clamp(image + mask.unsqueeze(0) * torch.tensor([0.45, 0.45, 0.42]).view(3, 1, 1), 0.0, 1.0)
        noise = torch.randn((3, size, size), generator=generator) * self.config.noise_std
        image = torch.clamp(image + noise, 0.0, 1.0)
        return image.float(), mask.unsqueeze(0).float()


def _sky_background(size: int, generator: torch.Generator) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, size).view(1, size, 1)
    blue = 0.72 + 0.18 * (1.0 - y)
    green = 0.48 + 0.10 * (1.0 - y)
    red = 0.32 + 0.08 * (1.0 - y)
    image = torch.cat([red.expand(1, size, size), green.expand(1, size, size), blue.expand(1, size, size)], dim=0)
    tint = torch.rand((3, 1, 1), generator=generator) * 0.04 - 0.02
    return torch.clamp(image + tint, 0.0, 1.0)


def _random_line_or_curve_mask(size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    horizontal = bool(torch.randint(0, 2, (1,), generator=generator).item())
    slope = float(torch.empty(1).uniform_(-0.45, 0.45, generator=generator).item())
    intercept = float(torch.empty(1).uniform_(0.15 * size, 0.85 * size, generator=generator).item())
    curve = float(torch.empty(1).uniform_(-0.0009, 0.0009, generator=generator).item())
    width = float(torch.empty(1).uniform_(1.2, 2.7, generator=generator).item())

    center = size / 2
    if horizontal:
        expected_y = intercept + slope * (xx - center) + curve * (xx - center).pow(2)
        distance = torch.abs(yy - expected_y)
    else:
        expected_x = intercept + slope * (yy - center) + curve * (yy - center).pow(2)
        distance = torch.abs(xx - expected_x)
    return (distance <= width).float()


def _cloudish_negative(size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    cloud = torch.zeros((size, size), dtype=torch.float32)
    blob_count = int(torch.randint(1, 4, (1,), generator=generator).item())
    for _ in range(blob_count):
        cx = float(torch.empty(1).uniform_(0, size, generator=generator).item())
        cy = float(torch.empty(1).uniform_(0, size, generator=generator).item())
        radius = float(torch.empty(1).uniform_(12, 32, generator=generator).item())
        distance2 = (xx - cx).pow(2) + (yy - cy).pow(2)
        cloud = torch.maximum(cloud, torch.exp(-distance2 / (2 * radius * radius)))
    return torch.clamp(cloud * (0.5 + 0.5 * math.sin(self_seed_safe(generator))), 0.0, 1.0)


def self_seed_safe(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator).item() * math.tau)
