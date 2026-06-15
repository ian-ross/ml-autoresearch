# single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es

Single-Frame RGB U-Net Candidate Experiment for Ground-Camera Contrail Detection. This is a controlled training-policy probe of the current best documented p=0.075 extra-wide Line Target auxiliary architecture using Harness-owned reduce-on-plateau scheduling and early stopping.

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
- `training.max_epochs: 80`
- `training.scheduler.policy: reduce_on_plateau` with factor `0.5`, patience `5`, and minimum LR `1e-5`
- `training.early_stopping`: enabled with patience `12`, min delta `0.001`, and best-checkpoint restoration enabled

## Contract assumptions

Candidate code defines only the Model Architecture. The Harness owns data loading, target derivation, losses, optimizer construction, scheduler behavior, early-stopping decisions, validation metric selection, checkpoint restoration, training, validation, artifact writing, and resource policy. No filesystem access, networking, runtime weight downloads, custom training loops, custom losses, or custom data transforms are used.

## Known limitations

The only intended scientific change from `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` is the Harness-owned training policy. Because the latest completed head-dropout Run was not observable in recent Agent Control Boundary reports, this candidate uses the best documented and evaluated Result, `run_20260602_203450_c05550`, as its primary comparison target.
