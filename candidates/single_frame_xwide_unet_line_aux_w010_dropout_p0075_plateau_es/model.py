"""Extra-wide U-Net-style Candidate Experiment with p=0.075 bottleneck dropout and line auxiliary logits for scheduler/early-stopping training."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolution blocks used throughout the U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pooling downsample followed by local feature extraction."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample, concatenate skip features, then refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class ExtraWideLightDropoutUNet(nn.Module):
    """Base-64 encoder/decoder with lighter bottleneck dropout and mask/line heads."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = Down(base_channels, base_channels * 2)
        self.enc3 = Down(base_channels * 2, base_channels * 4)
        self.bottleneck = Down(base_channels * 4, base_channels * 8)
        self.bottleneck_dropout = nn.Dropout2d(p=0.075)
        self.up3 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = Up(base_channels * 2, base_channels, base_channels)
        self.mask_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.line_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.bottleneck_dropout(self.bottleneck(s3))
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        features = self.up1(x, s1)
        return {"mask_logits": self.mask_head(features), "line_logits": self.line_head(features)}


def build_model(input_spec: dict, output_spec: dict) -> nn.Module:
    if input_spec["mode"] != "single_frame_rgb":
        raise ValueError("single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es requires single_frame_rgb input")
    if output_spec["form"] != "mask_logits":
        raise ValueError("single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es produces mask_logits output")
    auxiliary_outputs = output_spec.get("auxiliary_outputs", [])
    expected_auxiliary = [{"target": "line", "name": "line_logits", "shape": [1, 128, 128]}]
    if auxiliary_outputs != expected_auxiliary:
        raise ValueError("single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es requires line_logits auxiliary output")
    input_shape = input_spec.get("shape", [3, 128, 128])
    output_shape = output_spec.get("shape", [1, 128, 128])
    return ExtraWideLightDropoutUNet(in_channels=int(input_shape[0]), out_channels=int(output_shape[0]), base_channels=64)
