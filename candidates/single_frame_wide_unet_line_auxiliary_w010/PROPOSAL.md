# Experiment Proposal: wide-capacity Line Target auxiliary U-Net

## Hypothesis

Increasing the current best large Line Target auxiliary U-Net from base 32 to base 48 will test whether capacity scaling remains beneficial under the current Candidate Experiment Contract. The base-32 model improved best-validation Dice to 0.8158022490631504 over the base-24 model, suggesting this architecture family has not yet saturated. A wider model may better represent faint, broken, or thin contrails while preserving the Line Target auxiliary signal that has consistently helped this family.

## Comparison Target

Primary Comparison Target: `run_20260530_073538_f7c8b7` / `single_frame_large_unet_line_auxiliary_w010`, with best-validation `val/dice` 0.8158022490631504, `val/iou` 0.6889071089839139, `val/precision` 0.8145274118699493, and `val/recall` 0.8170810830709292.

Secondary context: `run_20260529_155844_d8ebec` showed base-24 capacity improved over earlier small variants, while `run_20260529_210959_8b9a18` and `run_20260530_043004_3e7aca` showed auxiliary-weight reduction and light photometric augmentation did not beat the base-24 unaugmented setting. This supports one more bounded capacity-scaling probe before returning to other contract features.

## Expected Effect

The base-48 U-Net should provide substantially more channels in encoder, bottleneck, and decoder blocks, improving recall and mask continuity on subtle contrails without changing data policy or loss semantics. Expected improvement is a modest best-validation Dice gain over the base-32 comparison target. Risks are overfitting, slower training, or Resource Failure from higher memory use.

## Implementation Sketch

Implement the same three-downsample U-Net topology as `single_frame_large_unet_line_auxiliary_w010`, but increase `base_channels` from 32 to 48. Keep the shared decoder and two 1x1 heads: `mask_logits` for the primary Contrail Mask and `line_logits` for the Harness-derived Line Target. Keep `input_mode: single_frame_rgb`, `output_form: mask_logits`, `data.sampling_policy: deterministic_shuffle`, `data.augmentation_policy: none`, primary `bce_dice`, auxiliary `weighted_bce` with weight 0.10, AdamW learning rate 0.001, and max epochs 30. Request batch size 12 to reduce memory pressure while staying within Harness-owned training knobs; the Harness may still lower the effective batch size if needed.

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

One standard GVCCS training Run under the existing wall-clock/resource policy: requested batch size 12, max epochs 30, no Experiment Batch, no Post-Run Evaluation requested in this proposal.

## Success Criteria

Consider the hypothesis supported if the completed Run improves best-validation `val/dice` over 0.8158022490631504 by at least 0.002 without severe precision/recall collapse or Resource Failure. Also inspect final/best metric separation and prediction samples for overfitting or obvious calibration pathology.

## Fallback/Next Decision

If the Run completes but fails to improve, treat simple U-Net capacity scaling as likely saturated under the current contract and pivot to diagnostics, threshold behavior, or a different in-contract architecture variation. If it improves, promote it as the new architecture baseline and request bounded diagnostics on the new best Run. If it fails from resource pressure, classify it as a Resource Failure and avoid covert memory workarounds in candidate code.
