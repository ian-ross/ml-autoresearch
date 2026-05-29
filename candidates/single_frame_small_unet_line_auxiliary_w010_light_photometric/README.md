# single_frame_small_unet_line_auxiliary_w010_light_photometric

Candidate Experiment for Ground-Camera Contrail Detection using the established base-16 single-frame Small U-Net with primary mask logits and a Harness-derived Line Target auxiliary head at weight `0.10`.

This variant isolates the Harness-owned `light_photometric` Augmentation Policy after both `light_combined` and `light_geometric` augmentation regressed relative to the unaugmented lower-weight Line Target auxiliary Small U-Net. It keeps architecture, losses, optimizer, learning rate, batch size, max epochs, deterministic shuffle, and auxiliary weight unchanged relative to `single_frame_small_unet_line_auxiliary_w010`.

## Model Architecture

- Encoder/decoder Small U-Net with three downsampling stages, bilinear upsampling, skip connections, and `DoubleConv` blocks.
- Shared decoder features feed two `1x1` heads:
  - `mask_logits` for the Contrail Mask.
  - `line_logits` for the Harness-derived Line Target auxiliary loss.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Line Target auxiliary output `line_logits` with `weighted_bce` weight `0.10`
- Primary loss `bce_dice`
- Optimizer `adamw`
- Learning rate `0.001`
- Batch size `16`
- Max epochs `30`
- `data.sampling_policy: deterministic_shuffle`
- `data.augmentation_policy: light_photometric`

## Contract assumptions and limitations

The Candidate Experiment relies only on Harness-owned data loading, augmentation, losses, optimization, training, validation, and artifact persistence. It does not include custom data loaders, custom transforms, custom losses, custom training loops, filesystem access, network access, pretrained weights, or runtime downloads.
