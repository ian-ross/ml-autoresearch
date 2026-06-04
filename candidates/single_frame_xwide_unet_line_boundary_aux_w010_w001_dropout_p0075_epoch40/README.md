# single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40

Extra-wide single-frame U-Net Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

The model is a base-64 U-Net-style encoder/decoder with three downsampling stages, bilinear upsampling, skip connections, BatchNorm/ReLU double-convolution blocks, and `Dropout2d(p=0.075)` at the bottleneck. It returns three image-aligned heads from the final decoder feature map:

- `mask_logits` for the primary Contrail Mask;
- `line_logits` for the Harness-derived Line Target auxiliary loss;
- `boundary_logits` for the Harness-derived Boundary Target auxiliary loss.

## Manifest choices

- Input mode: `single_frame_rgb`.
- Output form: `mask_logits`.
- Auxiliary targets: line `weighted_bce` at weight 0.10 and boundary `weighted_bce` at very low weight 0.01.
- Data policy: deterministic shuffle sampling, no augmentation.
- Training: `bce_dice`, AdamW, learning rate 0.001, batch size 8, max epochs 40.

## Contract assumptions

Boundary and Line Target tensors and losses are Harness-owned. Candidate code does not load data, derive targets, implement losses, inspect files, write artifacts, use networking, or alter the training loop.

## Known limitations

This is a narrow Boundary Target weight ablation. Prior boundary weight 0.03 increased precision but hurt recall; this candidate intentionally uses one-third of that boundary weight to test whether edge supervision can act as a small regularizer without dominating the primary mask objective.
