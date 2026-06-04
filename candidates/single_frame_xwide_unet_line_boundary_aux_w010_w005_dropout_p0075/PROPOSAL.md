# Experiment Proposal: add Boundary Target auxiliary supervision to the p=0.075 extra-wide Line Target U-Net

## Hypothesis

`run_20260601_085755_25cd06` is the current best in-contract Result: the extra-wide base-64 U-Net with Line Target auxiliary weight `0.10` and lighter bottleneck `Dropout2d(p=0.075)` improved best-validation Dice to `0.834603`, raised recall to `0.822995`, and reduced aggregate empty-mask false positives in the whole-validation failure-bucket review. Its remaining qualitative risks are boundary under-segmentation on large positives, broad false-positive-heavy masks, and tiny missed positives.

Adding the now-implemented Harness-derived Boundary Target auxiliary output with a conservative `weighted_bce` weight `0.05` may sharpen decoder features around mask edges and reduce boundary under-segmentation/over-extension while preserving the p=0.075 model's recall gain. Keeping the existing Line Target auxiliary weight `0.10` tests whether centerline and edge cues are complementary under the same architecture and training policy.

## Comparison Target

Primary comparison target: `run_20260601_085755_25cd06` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`, especially best-validation `val/dice=0.834603`, precision `0.846543`, recall `0.822995`, and the failure-bucket evaluation `eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets`.

Secondary context: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`, which has slightly lower Dice and recall but higher precision, plus the notes showing that pure dropout-rate tuning has reached a narrow frontier.

## Expected Effect

Expected effect is a small improvement in precision or IoU at similar recall by making shared decoder features more edge-aware. A successful Run should maintain the p=0.075 recall improvement while reducing false-positive-heavy spillover and false-negative boundary erosion in later failure-bucket evaluation. The main risk is that the additional boundary auxiliary loss over-constrains the shared representation, reduces recall on thin/tiny positives, or adds negligible signal beyond the Line Target.

## Implementation Sketch

Start from the accepted p=0.075 extra-wide U-Net architecture: base width 64, three skip-connected decoder stages, bottleneck-only `Dropout2d(p=0.075)`, and a `mask_logits` head plus `line_logits` head from final decoder features. Add one parallel `boundary_logits` 1x1 convolution head from the same final decoder features. Declare both Harness-derived auxiliary targets in the manifest: Line Target `weighted_bce` weight `0.10` and Boundary Target `weighted_bce` weight `0.05`. Keep input mode, primary output, data policy, loss, optimizer, learning rate, batch size, and epoch budget identical to the comparison target.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-derived Boundary Target auxiliary output (`boundary_logits`) with `weighted_bce` weight `0.05`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Candidate code limited to Model Architecture; no custom data loading, target derivation, losses, training loop, filesystem access, networking, or artifact writing.

## Budget Requested

- One GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8`, matching the comparison target to avoid new resource pressure.
- `max_epochs: 30`.
- Expected parameter count is effectively unchanged from the p=0.075 comparison except for one extra 1x1 auxiliary head, remaining well below the 10M parameter budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure or Harness interruption.
- Best-validation `val/dice` is at least within `0.0015` of the comparison target (`>=0.8331`) and ideally exceeds `0.834603`.
- Precision improves over the comparison target or remains at least `0.845` while recall remains above `0.818`.
- Best-to-final Dice gap is not materially worse than the p=0.075 comparison, whose best epoch was also final epoch 30.
- If the Run is competitive, a follow-up failure-bucket Post-Run Evaluation should check boundary-heavy false negatives, false-positive-heavy spillover, empty-mask false positives, and missed tiny positives before promotion.

## Fallback/Next Decision

If Boundary Target auxiliary supervision improves Dice or precision at similar recall, promote it only after a bounded failure-bucket Post-Run Evaluation confirms that qualitative boundary and spillover failures improved. If it regresses recall or Dice, treat the result as evidence that the current one-pixel Boundary Target or weight is not yet beneficial with Line Target `0.10`; consider a smaller boundary weight, a line-vs-boundary ablation, or a human-gated loss/scheduler capability only if the failure pattern justifies it. If it fails due to candidate bug or contract issue, create a bounded Repair Candidate preserving this hypothesis and comparison target.
