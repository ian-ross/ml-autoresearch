# single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40

Extra-wide Single-Frame RGB U-Net Candidate Experiment for Ground-Camera Contrail Detection. This is a controlled training-budget probe of the current best p=0.075 bottleneck-dropout Line Target auxiliary model.

## Model Architecture

- Base-64 U-Net encoder/decoder with three skip-connected decoder stages.
- `Dropout2d(p=0.075)` after the bottleneck block.
- Shared final decoder features feeding two 1x1 heads:
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
- `training.max_epochs: 40`

## Contract assumptions

Candidate code defines only the Model Architecture. The Harness owns data loading, target derivation, losses, optimizer construction, training, validation, artifact writing, and resource policy. No filesystem access, networking, runtime weight downloads, custom training loops, custom losses, or custom data transforms are used.

## Known limitations

The only intended scientific change from `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun` is the longer 40-epoch budget. Any improvement must be checked for final-vs-best stability and possible precision loss before launching additional architecture changes.
