# single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine

Extra-wide single-frame U-Net Candidate Experiment for Ground-Camera Contrail Detection. It preserves the current best p=0.075 bottleneck dropout and Harness-derived Line Target auxiliary supervision, then adds a lightweight full-resolution residual refinement block before the mask and line heads.

## Model Architecture

- Single-frame RGB input.
- Base-64 U-Net encoder/decoder with three downsampling stages and bilinear upsampling.
- Bottleneck `Dropout2d(p=0.075)` as in the current best comparison target.
- Full-resolution residual `RefinementBlock` with dilation-1 and dilation-2 3x3 convolution branches fused by a 1x1 convolution.
- Two output heads: `mask_logits` for the primary Contrail Mask and `line_logits` for the Harness-derived Line Target auxiliary loss.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- `auxiliary_targets`: line / `line_logits` / `weighted_bce` / weight `0.10`
- `training.loss: bce_dice`
- `training.optimizer: adamw`
- learning rate `0.001`, batch size `8`, max epochs `30`
- `data.sampling_policy: deterministic_shuffle`
- `data.augmentation_policy: none`

## Contract assumptions and limitations

Candidate code only defines architecture modules and `build_model`. It does not load data, derive targets, implement losses, write artifacts, read the filesystem, access the network, or alter Harness-owned training behavior. Boundary Target auxiliary training is intentionally not used because recent runtime evidence shows that path is not safely executable yet.
