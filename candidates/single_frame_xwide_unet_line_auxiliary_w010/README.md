# single_frame_xwide_unet_line_auxiliary_w010

Extra-wide single-frame U-Net Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

- Single-frame RGB U-Net with three downsampling stages.
- Base channel width 64, doubling through the encoder to a 512-channel bottleneck.
- Bilinear upsampling decoder with skip concatenations.
- Shared final decoder features feed two 1x1 convolution heads: `mask_logits` and `line_logits`.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Line Target auxiliary output with `weighted_bce`, weight 0.10.
- `data.sampling_policy: deterministic_shuffle`
- `data.augmentation_policy: none`
- Primary loss `bce_dice`, optimizer `adamw`, learning rate 0.001.
- Requested batch size 8, max epochs 30.

## Contract assumptions

This Candidate Experiment only varies Model Architecture and allowed Harness-owned manifest choices. It does not implement data loading, transforms, losses, optimizer logic, training loops, artifact writing, filesystem inspection, networking, or pretrained weight access.

## Known limitations

The extra-wide model may overfit or consume more GPU memory than the base-48 comparison target. The requested lower batch size is intended to reduce resource pressure while keeping the experiment within allowed training knobs.
