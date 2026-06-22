# Experiment Proposal: temporal ASPP-context fine-scale refinement

## Hypothesis

The matched temporal-control diagnostics show that `run_20260618_204556_e6a60b` remains ahead of the single-frame temporal-eligible control mainly by missing fewer small positive masks and by slightly cleaner positive-mask predictions. Adding a small high-resolution residual refinement block after the final decoder stage should improve thin/faint contrail localization in the centered three-frame temporal ASPP-context family without the blunt recall loss observed from the earlier mask-gate variant.

## Comparison Target

Primary comparison target: `run_20260618_204556_e6a60b` / `temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context` on the same `centered_temporal_rgb_clip` input mode and `temporal_eligible_center` frame-selection policy.

Secondary reference: `run_20260620_092332_3ee9a1` / matched single-frame temporal-eligible ASPP-context control.

## Expected Effect

The fine-scale refinement should modestly increase best-validation Dice and IoU by recovering additional small positives while preserving the temporal run's precision. Expected evidence of success is fewer missed positive masks in a follow-up failure-bucket evaluation, with no broad increase in empty-mask false positives.

## Implementation Sketch

Start from the temporal ASPP-context U-Net. Keep the encoder, dilated bottleneck context, line auxiliary head, dropout, scheduler, early stopping, sampling policy, augmentation policy, and frame-selection policy unchanged. Add a lightweight `FineScaleRefinement` module after the last decoder block: parallel depthwise-separable 3x3 branches at dilation 1 and dilation 2, fused by a 1x1 convolution and added residually to the high-resolution decoder features before the mask and line heads. The block is architecture-only code and does not implement loss, data loading, transforms, or training logic.

## Contract Features Used

- `input_mode: centered_temporal_rgb_clip`
- `output_form: mask_logits`
- Harness-derived Line Target auxiliary output `line_logits` with `weighted_bce` at weight `0.10`
- Harness-owned `bce_dice` primary loss and `adamw` optimizer
- Harness-owned `deterministic_shuffle` sampling, `temporal_eligible_center` frame selection, and `none` augmentation
- Harness-owned `reduce_on_plateau` scheduler and enabled early stopping with best-checkpoint restoration
- Candidate code limited to Model Architecture modules in `model.py`

## Budget Requested

Same budget as the temporal ASPP-context parent: batch size 8, up to 80 epochs, reduce-on-plateau scheduler, early stopping patience 12, mixed precision by Harness default. The refinement adds only a small number of parameters and should remain below the 10M parameter smoke-test budget.

## Success Criteria

Strong success:

- Best `val/dice` exceeds `0.873707` by at least `0.0010` on the temporal-eligible validation subset.
- Best `val/precision` remains at least `0.8760`.
- Best `val/recall` remains at least `0.8690`.

Minimal useful success:

- Best `val/dice` matches or exceeds `0.873707` without reducing precision below `0.8750`, or
- A follow-up failure-bucket evaluation shows fewer fully missed positive masks than the temporal parent without increased empty-mask false-positive severity.

## Fallback/Next Decision

If Dice or precision regresses, treat this as evidence that high-resolution residual refinement overfits or over-expands thin structures and return to the unrefined temporal ASPP-context parent. If Dice improves but precision/empty-mask behavior is unclear, request a bounded failure-bucket Post-Run Evaluation before promoting. If the run fails for a candidate bug, create a Repair Candidate only if the fix preserves this hypothesis and comparison target.
