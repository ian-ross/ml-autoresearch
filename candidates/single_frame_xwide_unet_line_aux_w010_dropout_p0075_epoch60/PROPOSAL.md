# Experiment Proposal: extend recall-safer p=0.075 xwide Line Target U-Net to 60 epochs

## Hypothesis

The 40-epoch p=0.075 extra-wide U-Net with Line Target auxiliary weight 0.10 (`run_20260601_162117_66bf89`) remains the safer recall-sensitive design baseline: it reached its best validation Dice at epoch 40, preserved stronger default-threshold recall than the later p=0.05 leaderboard run, and its failure-bucket evaluation showed less precision bias. The p=0.05 40-epoch run (`run_20260602_101047_294476`) improved Dice by about 0.001 but did so by trading away recall. Extending the p=0.075 model to 60 epochs tests whether the recall-safer setting was still under-trained and can recover leaderboard Dice without the p=0.05 precision shift.

This is a narrow Harness-owned training-budget probe, not a new architecture or workaround. It avoids further Boundary Target tuning after very-low-weight boundary variants regressed, and avoids additional dropout lowering after p=0.05 produced a conservative operating point.

## Comparison Target

Primary comparison target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`, with best epoch 40, best-validation `val/dice=0.843361`, `val/iou=0.729148`, precision `0.855490`, recall `0.831572`, `val/loss=0.500193`, and `val/total_loss=0.510662`.

Global-best context: `run_20260602_101047_294476` / `single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40`, with best epoch 39, best-validation `val/dice=0.844391`, precision `0.866790`, and recall `0.823120`. The proposed candidate should report against this Result, but it is not the primary control because its failure-bucket Research Note concluded that it is a precision-biased leaderboard best rather than the safest general-purpose baseline.

## Expected Effect

If the p=0.075 model was still improving at epoch 40, a 60-epoch budget should produce a small improvement in best-validation Dice and IoU while preserving recall near or above the 40-epoch p=0.075 target. A useful outcome would close or exceed the p=0.05 Dice gap without increasing precision at the expense of missed positives. If the added epochs overfit, best-validation selection and final-vs-best metrics should reveal instability before this line is extended further.

The main risk is extra wall-clock use for a marginal gain or validation overfitting from repeated epoch-budget probes. This proposal bounds that risk by changing only `max_epochs` and by treating this as the final simple epoch-extension test before requesting scheduler/early-stopping capability or returning to architecture changes.

## Implementation Sketch

Reuse the accepted extra-wide base-64 U-Net with shared encoder/decoder, mask head, Line Target auxiliary head, and bottleneck `Dropout2d(p=0.075)`. Keep `input_mode`, outputs, Line Target weight, sampling policy, augmentation policy, loss, optimizer, learning rate, batch size, and architecture identical to `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`. Change only `training.max_epochs` from 40 to 60 in the manifest.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Harness-owned bounded training knob `max_epochs: 60`.
- Candidate code limited to Model Architecture; no custom data loading, training loop, loss, filesystem access, networking, runtime weight download, or artifact writing.

## Budget Requested

- One GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8`, matching the comparison target to avoid new resource pressure.
- `max_epochs: 60`, inside the documented contract bound of 1 to 100 epochs.
- Parameter count unchanged from the p=0.075 40-epoch base, approximately 7.8M and below the 10M parameter budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure or Harness interruption.
- Best-validation `val/dice` exceeds the primary comparison target (`>0.843361`) and ideally matches or exceeds the p=0.05 global-best context (`>=0.844391`).
- Recall remains near the p=0.075 comparison target (`>=0.829`) and does not fall to the p=0.05 precision-biased range.
- Precision remains within a useful range (`>=0.850`) without a broad false-positive regression.
- Final-vs-best Dice gap is no more than about `0.005`; a larger gap indicates the extra budget is less stable and should trigger scheduler/early-stopping discussion rather than more fixed-epoch tuning.

## Fallback/Next Decision

If 60 epochs improves Dice while preserving recall, promote it as the new recall-sensitive baseline and request a bounded failure-bucket Post-Run Evaluation before further candidates. If it improves Dice but becomes precision-biased like p=0.05, treat it as leaderboard context rather than a general-purpose baseline. If it regresses or shows an unstable final-vs-best gap, stop simple epoch extension and prefer a Capability Request for Harness-owned scheduler or early stopping support, or return to an in-contract decoder change aimed at missed positives.
