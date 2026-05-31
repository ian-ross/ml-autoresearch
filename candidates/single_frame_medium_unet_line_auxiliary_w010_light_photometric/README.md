# single_frame_medium_unet_line_auxiliary_w010_light_photometric

## Model Architecture

Medium-capacity single-frame U-Net for Ground-Camera Contrail Detection. The network uses a base width of 24 channels, three encoder downsampling stages, bilinear decoder upsampling with skip connections, and two 1x1 output heads:

- `mask_logits` for the primary Contrail Mask prediction.
- `line_logits` for the Harness-derived Line Target auxiliary loss.

The architecture is copied from the current best medium Line Target auxiliary family so this Candidate Experiment isolates the Harness-owned augmentation-policy change.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Line Target auxiliary output `line_logits`
- Primary loss: `bce_dice`
- Auxiliary loss: `weighted_bce`, weight `0.10`
- Optimizer: `adamw`
- Learning rate: `0.001`
- Batch size: `16`
- Max epochs: `30`
- Sampling policy: `deterministic_shuffle`
- Augmentation policy: `light_photometric`

## Contract assumptions

This Candidate Experiment relies only on allowlisted Candidate Experiment Contract features. Candidate code defines architecture modules and `build_model` only. Training, data loading, Line Target derivation, augmentation, loss computation, evaluation, artifact writing, and resource policy remain Harness-owned.

## Known limitations

The Candidate Experiment changes only photometric augmentation relative to `single_frame_medium_unet_line_auxiliary_w010`; it does not add temporal context, boundary targets, pretrained weights, custom losses, custom sampling, early stopping, or a scheduler. Recent medium-family Runs reached their best validation metrics before the final epoch, so the resulting Run should be interpreted using `best_metrics.json` as well as final metrics.
