# Experiment Proposal: high-resolution detail fusion for the 60-epoch p=0.075 xwide Line Target U-Net

## Hypothesis

`run_20260602_203450_c05550` is the current best in-contract Result, but its failure-bucket evaluation still shows fully missed tiny positive masks, broad false-positive spillover on some positive sequences, and false-negative-heavy under-segmentation of large positives. The existing extra-wide U-Net already has the right training budget and recall-sensitive `Dropout2d(p=0.075)` setting, so the next useful test is a small architecture change rather than another fixed-epoch extension.

A shallow high-resolution detail-fusion head should help the final mask and Line Target heads use the earliest encoder features more directly after the last decoder stage. The expected benefit is better preservation of thin, low-area positive evidence and sharper large-mask completion while keeping the proven encoder/decoder, line auxiliary target, dropout, optimizer, data policy, and 60-epoch budget unchanged.

## Comparison Target

Primary comparison target: `run_20260602_203450_c05550` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`, with best-validation `val/dice=0.849036`, `val/iou=0.737673`, precision `0.867065`, recall `0.831741`, and failure-bucket evaluation Dice `0.849040` at threshold 0.5.

The next Research Note should also report the failure-bucket caution from that comparison: 74 missed-positive validation samples, 1,107 false-positive pixels on empty masks, and 184 empty-mask samples with any false positive.

## Expected Effect

If missed tiny masks and under-segmented large masks are partly caused by decoder features losing high-resolution detail, fusing a compact projection of the first encoder block with the final decoder features should increase recall or Dice without materially increasing parameter count. A useful result would improve best-validation Dice above `0.849036`, preserve recall at or above `0.831`, and avoid worsening the empty-mask affected-sample count in subsequent failure-bucket evaluation.

The main risk is that exposing more high-resolution texture will increase empty-mask false positives or broaden spillover. That risk is bounded by making the detail branch small and keeping the same dropout, Line Target weight, deterministic shuffle data policy, no augmentation, and training budget.

## Implementation Sketch

Start from `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`. Keep the base-64 U-Net encoder/decoder, bottleneck `Dropout2d(p=0.075)`, and shared mask/line auxiliary heads. Add an architecture-only detail fusion module after the final decoder stage:

- project the first encoder feature map (`s1`, full resolution) from 64 to 32 channels with a 3x3 convolution, batch normalization, and ReLU;
- concatenate that compact detail tensor with the final decoder tensor;
- run a two-convolution refinement block back to 64 channels;
- feed both `mask_logits` and `line_logits` heads from the refined features.

No data loading, custom loss, custom target derivation, runtime filesystem access, artifact writing, or training-loop changes are introduced.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Harness-owned bounded training knobs: learning rate `0.001`, batch size `8`, max epochs `60`.
- Candidate code limited to Model Architecture; no custom data loading, training loop, loss, filesystem access, networking, runtime weight download, or artifact writing.

## Budget Requested

- One GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8` and `max_epochs: 60`, matching the current best comparison target.
- Expected parameter count remains below the 10M smoke-test budget because the added detail projection/refinement is small relative to the 7.8M-parameter baseline.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure or Harness interruption.
- Best-validation `val/dice` exceeds the primary comparison target (`>0.849036`) or improves recall-sensitive behavior enough to justify diagnostic follow-up.
- Best-validation recall remains at least `0.831` and precision remains at least `0.860`.
- Final-vs-best Dice gap stays no larger than about `0.005`.
- If promoted to failure-bucket evaluation, it should not materially worsen empty-mask affected-sample count relative to 184 / 1,004 while trying to reduce missed-positive samples below 74 / 2,885.

## Fallback/Next Decision

If the detail-fusion head improves Dice and recall without obvious precision collapse, request a bounded failure-bucket Post-Run Evaluation comparing it to `run_20260602_203450_c05550`. If it improves Run-level Dice but worsens empty-sky cleanliness, treat it as a diagnostic architecture variant rather than the new default. If it regresses, abandon this high-resolution fusion path and prefer either a different small architecture change or a Capability Request for Harness-owned scheduler/early-stopping support rather than more fixed-epoch tuning.
