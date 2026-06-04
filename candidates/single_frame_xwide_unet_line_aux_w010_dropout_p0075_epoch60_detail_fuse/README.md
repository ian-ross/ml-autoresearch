# single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse

Extra-wide Single-Frame RGB U-Net Candidate Experiment for Ground-Camera Contrail Detection. This variant starts from the current best 60-epoch p=0.075 Line Target auxiliary baseline and adds a small high-resolution detail-fusion head.

## Model Architecture

- Base-64 U-Net encoder/decoder with three skip-connected decoder stages.
- `Dropout2d(p=0.075)` after the bottleneck block.
- Compact full-resolution detail branch from the first encoder feature map:
  - 3x3 convolution projects `s1` from 64 to 32 channels.
  - Projected detail features concatenate with the final decoder features.
  - A two-convolution refinement block returns fused features to 64 channels.
- Shared fused features feed two 1x1 heads:
  - `mask_logits` for the primary Contrail Mask.
  - `line_logits` for the Harness-derived Line Target auxiliary loss.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- `auxiliary_targets`: Line Target with `weighted_bce`, weight `0.10`
- `data.sampling_policy: deterministic_shuffle`
- `data.augmentation_policy: none`
- `training.loss: bce_dice`
- `training.optimizer: adamw`
- `training.learning_rate: 0.001`
- `training.batch_size: 8`
- `training.max_epochs: 60`

## Contract assumptions

Candidate code defines only the Model Architecture. The Harness owns data loading, target derivation, losses, optimizer construction, training, validation, artifact writing, and resource policy. No filesystem access, networking, runtime weight downloads, custom training loops, custom losses, or custom data transforms are used.

## Known limitations

The architectural change may expose high-resolution texture that increases empty-mask false positives. The next Research Note should compare both aggregate best-validation metrics and, if evaluated, failure-bucket missed-positive and empty-mask affected-sample counts against `run_20260602_203450_c05550`.
