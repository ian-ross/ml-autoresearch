# Experiment Proposal: single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40

## Hypothesis

The current best in-contract Result, `run_20260601_162117_66bf89` (`single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`), improved both precision and recall at 40 epochs with `Dropout2d(p=0.075)`, but the failure-bucket evaluation still shows missed tiny positives and false-negative-heavy large positives. Reducing only the bottleneck dropout from `p=0.075` to `p=0.05` may preserve the regularization that helped the extra-wide U-Net while slightly increasing feature capacity for faint or under-segmented contrails.

## Comparison Target

Primary comparison target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`.

Secondary context:

- `run_20260601_085755_25cd06` / 30-epoch p=0.075 model.
- `run_20260530_180658_0af8a8` / 30-epoch p=0.10 model.
- `run_20260531_160756_9f5a12` / 30-epoch p=0.15 model, which regressed.

## Expected Effect

Expected primary effect is a small recall and Dice improvement relative to the 40-epoch p=0.075 model, especially on missed-positive and false-negative-heavy validation cases. The main risk is that weaker dropout increases broad false positives or empty-mask false-positive pixels. Precision should therefore be treated as a guardrail, not just an auxiliary metric.

## Implementation Sketch

Start from the current best base-64 extra-wide single-frame U-Net with Line Target auxiliary output and 40 epochs. Keep the encoder, decoder, line auxiliary head, Harness-owned data policy, loss, optimizer, learning rate, batch size, and training budget unchanged. Change only the bottleneck `Dropout2d` probability from `0.075` to `0.05`.

The Candidate Experiment code remains architecture-only. It does not implement losses, target derivation, data loading, training loops, evaluation, filesystem access, networking, or artifact writes.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Primary `mask_logits` output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-owned deterministic shuffle Sampling Policy.
- Harness-owned no-augmentation policy.
- Harness-owned `bce_dice` primary loss.
- Harness-owned AdamW optimizer with learning rate `0.001`.
- Harness-owned training knobs: batch size `8`, max epochs `40`.
- Model Architecture change only: bottleneck dropout probability `0.05`.

## Budget Requested

One GVCCS Working Validation Split Run with the same resource profile as the current best 40-epoch extra-wide U-Net. Expected parameter count remains about 7.8M, below the 10M parameter budget. No Post-Run Evaluation is requested in this Autonomy Step; if the Run improves or creates an ambiguous precision/recall tradeoff, request a bounded failure-bucket review later.

## Success Criteria

Promote the Candidate Experiment only if it satisfies all of the following against `run_20260601_162117_66bf89`:

- Best-validation `val/dice` improves by at least `+0.0005`.
- Best-validation `val/recall` improves or stays within `0.002` of the comparison target.
- Best-validation `val/precision` does not decrease by more than `0.005`.
- No Resource Failure or contract violation occurs.
- Qualitative first-N prediction samples do not show obvious broad false-positive spillover.

If the best-validation Dice is near-tied but precision falls, do not promote without a later failure-bucket Evaluation Request focused on empty-mask false positives and false-positive-heavy positives.

## Fallback/Next Decision

If p=0.05 regresses or increases false positives, keep `run_20260601_162117_66bf89` as the current best and treat p=0.075 as the preferred dropout setting for this architecture. Do not continue monotonic dropout lowering unless the result specifically shows recall gains with controlled precision. If the result improves, request a bounded failure-bucket Post-Run Evaluation before using it as the next comparison target.
