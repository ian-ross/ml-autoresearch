# Experiment Proposal: very-low-weight Boundary Target auxiliary on p=0.075 epoch-40 xwide U-Net

## Hypothesis

The completed Boundary Target auxiliary run at weight 0.03 (`run_20260601_230602_718182`) trained successfully but shifted the current p=0.075 40-epoch extra-wide U-Net too far toward high precision and low recall. The whole-validation diagnostics for the p=0.05 dropout Result (`run_20260602_101047_294476`) also show that the current frontier is precision-biased and still vulnerable to tiny missed positives and large-mask under-segmentation. A much smaller Boundary Target auxiliary weight (`0.01`) may provide weak edge-local regularization while preserving the recall and Dice of the safer p=0.075 40-epoch baseline.

This is not another attempt to emulate unavailable supervision in candidate code: Boundary Target derivation and loss computation are requested only through the implemented Harness-owned auxiliary-target contract.

## Comparison Target

Primary comparison target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`, with best-validation `val/dice=0.843361`, `val/iou=0.729148`, precision `0.855490`, recall `0.831572`, and failure-bucket evaluation showing 1,210 empty-mask false-positive pixels, 143 empty-mask samples with any false positives, and 83 missed-positive samples.

Secondary context: `run_20260601_230602_718182` / boundary weight 0.03 regressed to best-validation `val/dice=0.841019` with precision `0.873355` and recall `0.810993`, showing the boundary effect was too strong at 0.03. `run_20260602_101047_294476` / p=0.05 dropout is the leaderboard best by Dice (`0.844391`) but is precision-biased and lower-recall than the p=0.075 baseline.

## Expected Effect

The expected effect is a small reduction in broad spillover or boundary fuzz without the recall collapse observed at boundary weight 0.03. If the boundary signal is useful only at higher weight, this variant may be indistinguishable from the p=0.075 baseline. If even weight 0.01 pushes the model toward lower recall, then Boundary Target auxiliary supervision is probably not the right immediate path for the current Line Target U-Net family at the default threshold.

## Implementation Sketch

Reuse the accepted extra-wide base-64 U-Net architecture from `single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40`: shared encoder/decoder, bottleneck `Dropout2d(p=0.075)`, and three 1x1 heads for `mask_logits`, `line_logits`, and `boundary_logits`. Keep Single-Frame RGB Input, Line Target auxiliary weight 0.10, deterministic shuffle sampling, no augmentation, `bce_dice`, AdamW, learning rate 0.001, batch size 8, and `max_epochs: 40`. Change only the Boundary Target auxiliary weight from 0.03 to 0.01.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-derived Boundary Target auxiliary output (`boundary_logits`) with `weighted_bce` weight `0.01`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Harness-owned bounded training knobs: learning rate `0.001`, batch size `8`, max epochs `40`.
- Candidate code limited to Model Architecture; no custom data loading, target derivation, losses, filesystem access, networking, or artifact writing.

## Budget Requested

- One GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8`, matching the comparison target and prior boundary run.
- `max_epochs: 40`, matching the primary comparison target.
- Parameter count should remain approximately 7.786M, adding only the existing one-channel boundary head relative to the non-boundary p=0.075 baseline and staying below the 10M parameter budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure or Harness interruption.
- Best-validation `val/dice` matches or exceeds the p=0.075 baseline (`>=0.843361`) and ideally approaches or exceeds the p=0.05 leaderboard Result (`>=0.844391`).
- Recall remains within `0.005` of the p=0.075 baseline (`>=0.826572`), avoiding the boundary weight 0.03 failure mode.
- Precision does not fall by more than `0.005` versus the p=0.075 baseline (`>=0.850490`).
- If Dice improves or the precision/recall tradeoff changes materially, request a bounded failure-bucket Post-Run Evaluation before promoting the variant.

## Fallback/Next Decision

If boundary weight 0.01 improves or matches Dice while preserving recall, consider it a viable low-weight geometric regularizer and request failure-bucket diagnostics against `run_20260601_162117_66bf89` and `run_20260602_101047_294476`. If it regresses or remains precision-biased, stop Boundary Target auxiliary follow-up for this architecture family and prefer non-boundary, recall-oriented in-contract changes or a human-gated Capability Request for scheduler/early-stopping or additional loss support.
