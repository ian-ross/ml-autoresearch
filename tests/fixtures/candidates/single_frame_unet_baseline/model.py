"""Tiny PyTorch-looking Candidate Experiment fixture.

Issue #1 validates that this file exists and exposes build_model, but the
Harness does not import or execute it until the smoke-test issue.
"""

import torch
from torch import nn


class TinyMaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_spec, output_spec):
    if input_spec["mode"] != "single_frame_rgb":
        raise ValueError("single_frame_unet_baseline requires single_frame_rgb input")
    if output_spec["form"] != "mask_logits":
        raise ValueError("single_frame_unet_baseline produces mask_logits output")
    return TinyMaskModel()
