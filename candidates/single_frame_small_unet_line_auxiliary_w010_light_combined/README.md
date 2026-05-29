# single_frame_small_unet_line_auxiliary_w010_light_combined

Small U-Net Candidate Experiment that keeps the current best lower-weight Line Target auxiliary architecture and adds the Harness-owned `light_combined` Augmentation Policy.

Implementation summary:

- single-frame RGB input;
- base-16 encoder/decoder U-Net with skip connections;
- primary `mask_logits` output for Contrail Mask evaluation;
- auxiliary `line_logits` output trained against the Harness-derived Line Target;
- auxiliary `weighted_bce` weight `0.10`;
- `bce_dice` primary loss, AdamW learning rate `0.001`, batch size `16`, max epochs `30`;
- deterministic shuffled training order;
- `light_combined` Harness augmentation preset for training examples only.

The model code is architecture-only. Data loading, augmentation execution, target derivation, loss computation, training, evaluation, and artifact persistence remain Harness-owned.

Comparison target: `run_20260510_165335_b458c3` from `single_frame_small_unet_line_auxiliary_w010`.
