# Experiment Proposal: extra-wide Line Target U-Net with bottleneck dropout

## Hypothesis

The base-64 extra-wide Line Target auxiliary U-Net is the current best by best-validation Dice, but its diagnostics show a precision-biased default threshold, persistent small-positive misses, and a large final-vs-best degradation after epoch 28. Adding a small in-architecture bottleneck `Dropout2d(p=0.10)` may regularize the highest-level features enough to reduce late-epoch overconfidence and recover some recall without abandoning the precision and false-positive gains of the base-64 family.

## Comparison Target

Primary comparison target: `run_20260530_134005_6f20b1` / `single_frame_xwide_unet_line_auxiliary_w010`, especially best-validation `val/dice=0.832669` at epoch 28 and default-threshold recall `0.804909` from `eval_eval_2026_05_30_xwide_unet_failure_buckets`.

Secondary context: `run_20260530_101019_def893` / `single_frame_wide_unet_line_auxiliary_w010`, which had slightly lower Dice but better final-epoch stability than the xwide run.

## Expected Effect

Expected effect is a modest improvement in robustness rather than a large capacity gain: similar or slightly higher best-validation Dice, a smaller best-to-final degradation at epoch 30, and improved recall or fewer missed-positive masks at threshold 0.5. The risk is underfitting from dropout, which would show up as lower Dice and lower precision/recall than the xwide comparison.

## Implementation Sketch

Start from the accepted base-64 U-Net with shared encoder/decoder features, mask head, and Line Target auxiliary head. Insert only `nn.Dropout2d(p=0.10)` after the bottleneck block before the first upsampling block. Keep all data, loss, optimizer, auxiliary-target, and training settings identical to the xwide comparison target so the dropout change is isolated.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Candidate code limited to Model Architecture; no custom data loading, training loop, loss, filesystem access, networking, or artifact writing.

## Budget Requested

- GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8`, matching the xwide comparison to avoid new resource pressure.
- `max_epochs: 30`.
- Expected parameter count unchanged from the base-64 comparison and below the 10M parameter budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure.
- Best-validation `val/dice` is at least within `0.002` of the xwide comparison (`>=0.8307`) and ideally exceeds it.
- Recall at the best epoch improves over the xwide default-threshold recall (`>0.8049`) without dropping precision below the wide base-48 comparison (`0.8507`) by more than about `0.01`.
- Final epoch Dice is closer to best-validation Dice than the xwide comparison's roughly `0.024` gap.

## Fallback/Next Decision

If dropout improves or preserves best-validation Dice while reducing final-vs-best degradation, request a bounded failure-bucket Post-Run Evaluation to compare missed-positive and empty-mask false-positive behavior. If it regresses, abandon dropout regularization for this family and prefer a human-gated Capability Request for threshold selection, scheduler, or early-stopping support rather than further width-only scaling.
