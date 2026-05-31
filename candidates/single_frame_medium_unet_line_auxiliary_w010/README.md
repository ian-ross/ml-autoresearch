# single_frame_medium_unet_line_auxiliary_w010

Medium-capacity U-Net Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

This Candidate Experiment keeps the current best single-frame U-Net topology and Line Target auxiliary branch, but increases the channel width from base 16 to base 24. The model uses:

- three encoder downsampling stages plus a bottleneck;
- bilinear decoder upsampling with skip connections;
- a primary `mask_logits` 1x1 head for the Contrail Mask;
- an auxiliary `line_logits` 1x1 head for the Harness-derived Line Target.

## Manifest choices

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Line Target auxiliary output: `line_logits`, `weighted_bce`, weight `0.10`
- `data.sampling_policy: deterministic_shuffle`
- `data.augmentation_policy: none`
- primary loss: `bce_dice`
- optimizer: `adamw`
- learning rate: `0.001`
- batch size: `16`
- max epochs: `30`

## Contract assumptions

The Candidate Experiment relies only on Harness-owned data loading, augmentation policy, training, loss computation, metrics, resource policy, and artifact persistence. Candidate code defines Model Architecture only and does not access filesystems, networks, datasets, subprocesses, MLflow, Docker, or checkpoints.

## Known limitations

This is a capacity-scaling test, not a new data-policy or loss experiment. It may overfit or train more slowly than the base-16 comparison target. If it fails due to resource pressure, the follow-up should be handled through Run Failure Classification rather than by embedding resource-policy workarounds in model code.
