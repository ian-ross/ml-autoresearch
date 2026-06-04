# Experiment Proposal: extend current best p=0.075 xwide Line Target U-Net to 40 epochs

## Hypothesis

The current best in-contract Result, `run_20260601_085755_25cd06` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`, reached its best validation Dice at the final epoch 30 (`val/dice=0.834603`) with lower validation loss than the p=0.10 dropout base. This suggests the model may still benefit from a modestly longer Harness-owned training budget rather than another architecture change. Extending the same Candidate Experiment to `max_epochs: 40` may improve Dice or confirm that the 30-epoch result is already saturated while preserving the same default-threshold calibration and failure profile.

This proposal deliberately avoids another dropout-rate sweep, Boundary Target emulation, or full-resolution refinement. It tests one approved training knob on the current best base after the refinement variant regressed and after Boundary Target auxiliary training proved blocked by runtime contract support.

## Comparison Target

Primary comparison target: `run_20260601_085755_25cd06` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`, especially best-validation `val/dice=0.834603`, `val/iou=0.716153`, precision `0.846543`, recall `0.822995`, and final epoch equal to best epoch at epoch 30.

Secondary context: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout` and `run_20260601_122336_17faa3` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine`. The p=0.10 dropout model remains a high-precision reference, while the refinement model warns that adding architectural detail can improve first-N examples but hurt aggregate validation Dice and stability.

## Expected Effect

A 40-epoch budget should reveal whether the p=0.075 model was still on an improving trajectory at epoch 30. If the epoch-30 endpoint was not saturated, the expected effect is a small improvement in best-validation Dice and/or validation loss without a large precision drop. If later epochs overfit or drift toward broader masks, best-validation selection should still show whether any improvement occurs before final epoch, while final-vs-best gap will expose stability risk.

The main risk is spending more wall-clock on a narrow improvement or increasing validation overfitting pressure. This is bounded by changing only `max_epochs` and by stopping further local dropout tuning.

## Implementation Sketch

Reuse the accepted extra-wide base-64 U-Net with shared encoder/decoder, mask head, Line Target auxiliary head, and bottleneck `Dropout2d(p=0.075)`. Keep `input_mode`, outputs, Line Target weight, sampling policy, augmentation policy, loss, optimizer, learning rate, batch size, and architecture identical to `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`. Change only `training.max_epochs` from 30 to 40 in the manifest.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Harness-owned bounded training knob `max_epochs: 40`.
- Candidate code limited to Model Architecture; no custom data loading, training loop, loss, filesystem access, networking, or artifact writing.

## Budget Requested

- One GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8`, matching the current best comparison to avoid new resource pressure.
- `max_epochs: 40`, a one-third increase over the comparison target but still inside the documented contract bound.
- Parameter count unchanged from the current best p=0.075 base, approximately 7.8M and below the 10M parameter budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure or Harness interruption.
- Best-validation `val/dice` matches or exceeds the comparison target (`>=0.834603`) and ideally improves by at least `0.0005`.
- Precision remains within about `0.005` of the comparison target (`>=0.8415`) while recall remains above `0.820`.
- Final-vs-best Dice gap remains small (`<=0.005`) or the best epoch occurs late enough to suggest stable training rather than early overfit.
- Failure-bucket follow-up is only needed if best Dice improves or if the precision/recall tradeoff changes materially.

## Fallback/Next Decision

If 40 epochs improves best-validation Dice without a large precision or stability penalty, promote the longer-budget p=0.075 model as the new in-contract base and request a bounded failure-bucket Post-Run Evaluation before further candidate changes. If it regresses or only matches the 30-epoch base, keep `run_20260601_085755_25cd06` as current best and stop this local training-budget probe. If the longer run worsens final-vs-best stability, prefer a Harness-owned early-stopping or scheduler Capability Request rather than more manual epoch-count tuning.
