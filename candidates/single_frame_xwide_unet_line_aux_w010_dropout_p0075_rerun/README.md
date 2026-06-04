# single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun

Extra-wide base-64 U-Net Candidate Experiment for Ground-Camera Contrail Detection. It tests a lighter bottleneck dropout rate than the current best p=0.10 dropout model while keeping Line Target auxiliary supervision and all Harness-owned training choices fixed.

## Model Architecture

- Single-frame RGB encoder/decoder U-Net with base width 64.
- Four encoder stages including bottleneck, three bilinear decoder stages, and skip concatenations.
- `Dropout2d(p=0.075)` after the bottleneck block.
- Primary `mask_logits` head from final decoder features.
- Auxiliary `line_logits` head from the same final decoder features.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Line auxiliary target: `weighted_bce`, weight `0.10`
- `data.sampling_policy: deterministic_shuffle`
- `data.augmentation_policy: none`
- `training.loss: bce_dice`
- `training.optimizer: adamw`
- `learning_rate: 0.001`, `batch_size: 8`, `max_epochs: 30`

## Contract assumptions and limitations

The Candidate Experiment only defines Model Architecture code. The Harness owns data loading, Line Target derivation, losses, optimization, training, validation, artifact writing, and resource policy. No custom filesystem, networking, data-loading, target-derivation, loss, or training-loop authority is used.
