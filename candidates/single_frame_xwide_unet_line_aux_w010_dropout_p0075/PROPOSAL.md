# Experiment Proposal: extra-wide Line Target U-Net with lighter bottleneck dropout

## Hypothesis

The current best in-contract base, `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`, showed that bottleneck `Dropout2d(p=0.10)` improves recall and best-validation Dice over the plain extra-wide U-Net. Subsequent variants with attention gates or stronger dropout shifted too far toward recall and lost precision. A slightly lighter bottleneck dropout rate, `p=0.075`, may retain most of the useful regularization while recovering enough precision to improve the Dice tradeoff at the default threshold.

## Comparison Target

Primary comparison target: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`, especially best-validation `val/dice=0.833852`, precision `0.856300`, recall `0.812552`, and the failure-bucket evaluation at threshold 0.5 (`val/dice=0.833856`, precision `0.856288`, recall `0.812569`).

Secondary context: `run_20260530_134005_6f20b1` / `single_frame_xwide_unet_line_auxiliary_w010`, which had higher precision but lower recall, and `run_20260531_160756_9f5a12` / `single_frame_xwide_unet_line_aux_w010_dropout_p015`, which improved stability but regressed Dice by spending too much precision.

## Expected Effect

The expected effect is a small precision recovery relative to `p=0.10` while keeping recall above the plain xwide base. If the p=0.10 model is slightly over-regularized at the default threshold, p=0.075 should land between the plain xwide and p=0.10 variants and may improve best-validation Dice. The risk is that p=0.075 is too close to p=0.10 to matter, or that lower dropout reintroduces the plain xwide model's final-vs-best instability and missed-positive behavior.

## Implementation Sketch

Start from the accepted base-64 U-Net with shared encoder/decoder features, mask head, and Line Target auxiliary head. Change only the bottleneck dropout rate from `Dropout2d(p=0.10)` to `Dropout2d(p=0.075)`. Keep all manifest choices, data policy, loss, optimizer, learning rate, batch size, and epoch budget identical to the p=0.10 comparison target so the dropout-rate change is isolated.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Candidate code limited to Model Architecture; no custom data loading, training loop, loss, filesystem access, networking, or artifact writing.

## Budget Requested

- GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8`, matching the p=0.10 comparison to avoid new resource pressure.
- `max_epochs: 30`.
- Expected parameter count unchanged from the p=0.10 comparison and below the 10M parameter budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure.
- Best-validation `val/dice` is at least within `0.0015` of the p=0.10 comparison (`>=0.83235`) and ideally exceeds `0.833852`.
- Precision improves over or matches the p=0.10 comparison without dropping below the older wide base-48 comparison by more than about `0.005` (`>=0.8457`).
- Recall remains above the plain xwide base-64 comparison (`>0.8049`).
- Final-vs-best Dice gap is not materially worse than the p=0.10 gap of about `0.0138`.

## Fallback/Next Decision

If p=0.075 improves Dice or restores precision without losing the dropout recall gain, request a bounded failure-bucket Post-Run Evaluation before promotion. If it regresses or simply matches p=0.10 within noise, stop the bottleneck-dropout-rate sweep and pivot to a human-gated scheduler, early-stopping, threshold-selection, boundary-target, or additional-loss capability slice rather than continuing architecture-only tuning.
