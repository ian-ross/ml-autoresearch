# Experiment Proposal: medium-capacity Line Target auxiliary U-Net

## Hypothesis

A modest capacity increase from the current best base-16 Small U-Net to a base-24 U-Net will improve Contrail Mask Dice on the GVCCS Working Validation Split while preserving the useful lower-weight Line Target auxiliary signal. Recent augmentation variants showed that the current architecture is not helped by the available geometric-containing presets, while photometric-only augmentation was near-neutral; the next in-contract lever should therefore test whether representation capacity, not augmentation, is the limiting factor.

## Comparison Target

Primary Comparison Target: `run_20260510_165335_b458c3` / `single_frame_small_unet_line_auxiliary_w010`, with best-validation `val/dice` 0.7843769771696992.

Secondary context: `run_20260525_202245_c3bcd5` nearly matched that Result with `light_photometric` but did not improve it, and `light_geometric` / `light_combined` regressed substantially.

## Expected Effect

The base-24 model should have more encoder and decoder feature capacity for faint or thin contrails and mixed false-positive/false-negative residual modes, without changing the training loop, data access, loss implementation, or target semantics. Expected benefits are a small increase in best-validation Dice and a more favorable recall/precision balance than the current best. The main risk is overfitting or slower training from the larger architecture.

## Implementation Sketch

Implement the same U-Net topology as `single_frame_small_unet_line_auxiliary_w010`, but increase `base_channels` from 16 to 24. Keep the shared decoder and two 1x1 heads: `mask_logits` for the primary Contrail Mask and `line_logits` for the Harness-derived Line Target. Keep `input_mode: single_frame_rgb`, `output_form: mask_logits`, `data.sampling_policy: deterministic_shuffle`, `data.augmentation_policy: none`, primary `bce_dice`, auxiliary `weighted_bce` with weight 0.10, AdamW learning rate 0.001, batch size 16, and max epochs 30.

## Contract Features Used

- Model Architecture variation only in `model.py`.
- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Primary mask logits output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight 0.10.
- Harness-owned deterministic shuffle Sampling Policy.
- Harness-owned no-augmentation policy (`augmentation_policy: none`).
- Harness-owned `bce_dice` primary loss, `adamw` optimizer, learning rate, batch size, and epoch budget.

No custom data loading, transforms, training loop, loss implementation, filesystem access, network access, checkpoint download, or artifact writing is requested.

## Budget Requested

One standard GVCCS training Run under the existing wall-clock/resource policy: batch size 16, max epochs 30, no Experiment Batch, no Post-Run Evaluation requested in this proposal.

## Success Criteria

Consider the hypothesis supported if the completed Run improves best-validation `val/dice` over 0.7843769771696992 by at least 0.003 without a severe precision/recall collapse or resource failure. Also inspect final/best metric separation and first-N prediction samples for obvious overfitting or calibration pathology.

## Fallback/Next Decision

If the Run completes but fails to improve, treat it as evidence that simple capacity scaling is not enough for this family and avoid immediate larger-channel repeats. If it improves modestly, use it as the new capacity baseline and consider bounded diagnostics or a narrow training-knob follow-up. If it fails from resource pressure, classify as a Resource Failure and consider a smaller base-20 variant only if policy permits a bounded repair-like new proposal; do not hide resource controls in candidate code.
