# Experiment Proposal: large-capacity Line Target auxiliary U-Net

## Hypothesis

Increasing the current best medium Line Target auxiliary U-Net from base 24 to base 32 will test whether representation capacity remains a useful in-contract lever for Ground-Camera Contrail Detection. The base-24 model improved best-validation Dice over the prior base-16 model, while auxiliary-weight reduction and available augmentation presets did not improve it. A base-32 model may better represent faint or fragmented contrails without changing Harness-owned data, training, loss, or evaluation policy.

## Comparison Target

Primary Comparison Target: `run_20260529_155844_d8ebec` / `single_frame_medium_unet_line_auxiliary_w010`, with best-validation `val/dice` 0.7994525273696043, `val/precision` 0.8429952005805884, and `val/recall` 0.760187086855412.

Secondary context: `run_20260529_210959_8b9a18` lowered the Line Target auxiliary weight to 0.05 and regressed to best-validation Dice 0.7945989165999967; `run_20260530_043004_3e7aca` added `light_photometric` augmentation and regressed to best-validation Dice 0.7970676596635833. These Results suggest returning to unaugmented architecture exploration rather than further augmentation or auxiliary-weight changes.

## Expected Effect

The base-32 U-Net should provide more encoder, bottleneck, and decoder channels for fine contrail structure while preserving the Line Target auxiliary training signal that has been useful in the best-performing family. Expected improvement is a small best-validation Dice gain over the base-24 comparison target. The main risks are overfitting, slower training, or an even more conservative precision/recall balance if extra capacity sharpens only high-confidence predictions.

## Implementation Sketch

Implement the same three-downsample U-Net topology as `single_frame_medium_unet_line_auxiliary_w010`, but increase `base_channels` from 24 to 32. Keep the shared decoder and two 1x1 heads: `mask_logits` for the primary Contrail Mask and `line_logits` for the Harness-derived Line Target. Keep `input_mode: single_frame_rgb`, `output_form: mask_logits`, `data.sampling_policy: deterministic_shuffle`, `data.augmentation_policy: none`, primary `bce_dice`, auxiliary `weighted_bce` with weight 0.10, AdamW learning rate 0.001, batch size 16, and max epochs 30.

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

Consider the hypothesis supported if the completed Run improves best-validation `val/dice` over 0.7994525273696043 by at least 0.003 without a severe precision/recall collapse or Resource Failure. Also inspect final/best metric separation and first-N prediction samples for overfitting or obvious calibration pathology.

## Fallback/Next Decision

If the Run completes but fails to improve, treat simple capacity scaling beyond base 24 as no longer promising under the current contract and avoid immediate wider U-Net repeats. If it improves, promote it as the new architecture baseline and request or use bounded diagnostics to check whether recall-specific misses remain acceptable. If it fails from resource pressure, classify it as a Resource Failure and do not hide resource-policy workarounds in candidate code.
