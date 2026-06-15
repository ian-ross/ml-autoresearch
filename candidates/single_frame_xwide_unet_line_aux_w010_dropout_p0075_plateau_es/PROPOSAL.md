# Experiment Proposal: xwide p=0.075 U-Net with reduce-on-plateau and early stopping

## Hypothesis

The current best documented in-contract architecture, `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`, reached its best validation Dice near epoch 50 of a fixed 60-epoch run and showed some final-epoch calibration drift. A Harness-owned `reduce_on_plateau` scheduler plus validation-Dice early stopping with best-checkpoint restoration should preserve the same architecture's recall-sensitive behavior while reducing late-epoch instability and potentially improving best/final Dice without adding model complexity.

## Comparison Target

Primary comparison target: `run_20260602_203450_c05550` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`, the best documented Result with whole-validation `val/dice = 0.849040` at threshold 0.5 and run-level best `val/dice = 0.849036` at epoch 50.

Secondary context: `run_20260603_094446_05dea3` regressed after adding high-resolution detail fusion, and `run_20260603_164830_9fcbc4` is completed in the ledger but its artifacts were not observable in recent Agent Control Boundary reports. This proposal intentionally avoids new architectural changes and tests the now-available Harness-owned training-policy slice directly on the documented best architecture.

## Expected Effect

Expected improvements are modest but interpretable: equal or better best-validation Dice versus `run_20260602_203450_c05550`, a smaller best-to-final gap because the final restored model should correspond to the best validation checkpoint, and a precision/recall balance close to the p=0.075 60-epoch baseline. The scheduler may lower learning rate after validation plateaus, allowing useful refinement without the blunt overtraining risk of simply extending fixed epochs.

## Implementation Sketch

Reuse the exact Model Architecture from `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`: a base-64 single-frame U-Net with `Dropout2d(p=0.075)` at the bottleneck and two 1x1 heads for `mask_logits` and `line_logits`. Keep the Line Target auxiliary loss weight at 0.10, deterministic shuffle, no augmentation, AdamW, learning rate 0.001, and batch size 8. Increase the maximum budget to 80 epochs only as an upper bound, while selecting Harness-owned `reduce_on_plateau` (`factor: 0.5`, `patience: 5`, `min_lr: 1e-5`) and early stopping on working-validation `val/dice` (`patience: 12`, `min_delta: 0.001`, `restore_best_checkpoint: true`).

## Contract Features Used

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Harness-derived Line Target auxiliary output `line_logits` with `weighted_bce`, weight 0.10
- Harness-owned `bce_dice` primary loss
- Harness-owned `adamw` optimizer
- Harness-owned deterministic training sample shuffle
- Harness-owned `reduce_on_plateau` scheduler policy
- Harness-owned early stopping with best-checkpoint restoration
- No candidate-owned data loading, target derivation, losses, optimizer, scheduler, training loop, artifact writing, filesystem access, networking, or pretrained weights

## Budget Requested

One GVCCS training Run with batch size 8 and an 80-epoch maximum. Early stopping is expected to stop before the maximum if validation Dice plateaus. No Post-Run Evaluation is requested in this Candidate Submission; request failure-bucket review only if the Run matches or beats the documented best or exposes an unexpected precision/recall tradeoff.

## Success Criteria

Promote if the Run achieves best-validation `val/dice >= 0.8490` and the restored/final metrics do not show a materially worse precision/recall tradeoff than `run_20260602_203450_c05550`. Strong success is `val/dice >= 0.8510` or comparable Dice with fewer missed positives and no increase in empty-mask false-positive spread in a later failure-bucket evaluation. Treat as useful negative evidence if the scheduler/early-stopping policy does not improve best Dice or makes recall/precision materially worse.

## Fallback/Next Decision

If this regresses, keep `run_20260602_203450_c05550` as the preferred documented baseline and avoid further scheduler tweaks until a Research Note compares the resolved training policy and stop reason. If it improves or restores a cleaner final checkpoint, request a bounded failure-bucket evaluation to verify tiny-positive misses, broad spillover, large-positive under-segmentation, and empty-mask affected-sample counts before proposing further architecture changes.
