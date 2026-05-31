# Experiment Proposal: xwide U-Net with bottleneck dropout and decoder attention gates

## Hypothesis

The current best in-contract model, `single_frame_xwide_unet_line_aux_w010_dropout` (`run_20260530_180658_0af8a8`), improved recall and best-validation Dice but still misses small positive masks and remains false-negative-heavy on some larger low-contrast regions. Adding lightweight attention gates to decoder skip connections may suppress irrelevant sky/background skip features while preserving contrail-aligned detail, improving recall on difficult positive masks without a large empty-mask false-positive penalty.

## Comparison Target

Primary comparison target: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`.

Secondary context: `run_20260530_134005_6f20b1` / `single_frame_xwide_unet_line_auxiliary_w010` and `run_20260530_101019_def893` / `single_frame_wide_unet_line_auxiliary_w010`.

## Expected Effect

Expected best-validation effect is a small Dice improvement over the xwide dropout baseline, or at minimum a better recall/precision balance: fewer missed-positive and false-negative-heavy cases at threshold 0.5 while keeping precision near the xwide dropout baseline and avoiding empty-mask false positives worse than the earlier wide baseline. Because the gates are decoder-local and parameter-light, the change should stay below the 10M parameter budget and should not require a different training budget.

## Implementation Sketch

Start from the base-64 extra-wide U-Net with Line Target auxiliary output and bottleneck `Dropout2d(p=0.10)`. Replace each skip concatenation in the decoder with an additive attention gate: project the decoder gating tensor and skip tensor to an intermediate channel count, combine with ReLU, predict a one-channel sigmoid attention mask, and multiply the skip features before concatenation. The model still returns only `mask_logits` and `line_logits`; no custom loss, sampler, transform, training loop, data loading, filesystem access, or artifact writing is added.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Primary mask logits output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight 0.10.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy.
- No augmentation policy (`none`).
- Bounded training knobs: learning rate 0.001, batch size 8, max epochs 30.
- Architecture-only candidate code; no candidate-owned data loading, training loop, loss, transform, filesystem, network, pretrained weights, or side-channel outputs.

## Budget Requested

One standard GVCCS training Run with the same budget as the comparison target: batch size 8, maximum 30 epochs, CUDA if available, and the existing Harness resource policy. The expected parameter count is modestly above the 7.79M xwide dropout baseline but below the 10M contract budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure or contract violation.
- Best-validation `val/dice` exceeds `0.833856` from `run_20260530_180658_0af8a8`, or matches it within about 0.001 while improving `val/recall` without reducing `val/precision` below the older wide baseline (`0.850680`).
- Final-vs-best Dice gap does not materially exceed the xwide dropout gap, indicating gates did not worsen late-epoch instability.
- If followed by failure-bucket diagnostics, missed-positive and false-negative-heavy buckets improve without empty-mask false positives worse than the wide baseline.

## Fallback/Next Decision

If attention gates regress Dice or mainly increase false positives, do not continue adding decoder complexity. Treat the result as evidence that the current contract's architecture-only refinements are near saturation and prefer a Capability Request for a Harness-owned threshold selection, scheduler/early-stopping, boundary target, or additional loss slice. If the Run fails due to a candidate bug, submit at most one Repair Candidate preserving this hypothesis and Comparison Target.
