# single_frame_xwide_unet_line_aux_w010_dropout

Extra-wide single-frame U-Net for Ground-Camera Contrail Detection with the existing Harness-derived Line Target auxiliary head and a small bottleneck `Dropout2d(p=0.10)` regularizer.

## Model Architecture

- Base-64 U-Net encoder/decoder using bilinear upsampling and skip concatenation.
- Two-head output: `mask_logits` for the Contrail Mask and `line_logits` for the auxiliary Line Target.
- Dropout is applied only at the bottleneck after the deepest encoder block.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Line auxiliary target with `weighted_bce`, weight `0.10`
- deterministic shuffle sampling, no augmentation
- `bce_dice`, AdamW, learning rate `0.001`, batch size `8`, max epochs `30`

## Contract assumptions and limitations

Candidate code defines architecture only. It does not implement data loading, augmentation, losses, training loops, filesystem access, networking, or artifact writing. This Candidate Experiment isolates a bottleneck dropout regularization change relative to `single_frame_xwide_unet_line_auxiliary_w010`.
