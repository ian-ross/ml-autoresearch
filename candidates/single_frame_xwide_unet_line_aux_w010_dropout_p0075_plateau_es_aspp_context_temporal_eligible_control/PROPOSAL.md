# Experiment Proposal: matched single-frame ASPP-context temporal-eligible control

## Hypothesis

The apparent gain from `run_20260618_204556_e6a60b` may combine two effects: centered three-frame temporal RGB input and the narrower `temporal_eligible_center` frame-selection subset. A matched single-frame ASPP-context control on the same temporal-eligible center Target Frames will isolate the data-policy contribution. If this control approaches the temporal Run's Dice, much of the gain is likely due to excluding sequence-boundary frames; if it remains closer to the all-target-frame single-frame ASPP-context Run, the temporal channels are likely adding useful information.

## Comparison Target

Primary Comparison Target: `run_20260618_204556_e6a60b` / `temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`, because this candidate is a matched data-policy control for that temporal result.

Secondary reference: `run_20260615_140810_2bee94` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`, the same single-frame architecture trained on `all_target_frames`.

## Expected Effect

Expected outcome is a single-frame Result on the same temporal-eligible train/validation subset as the temporal candidate. It may slightly improve over the all-target-frame single-frame ASPP-context Run if sequence-boundary frames are harder or noisier, but should underperform the temporal Run if neighboring-frame context is the main source of the recent improvement.

## Implementation Sketch

Reuse the single-frame extra-wide base-64 U-Net with p=0.075 bottleneck dropout, Line Target auxiliary head, and lightweight dilated bottleneck context from `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`. Keep the architecture and training policy unchanged. Change only the Harness-owned data policy by explicitly setting `data.frame_selection_policy: temporal_eligible_center` while keeping `input_mode: single_frame_rgb`.

## Contract Features Used

- `single_frame_rgb` Input Mode.
- `mask_logits` primary output.
- Harness-derived Line Target auxiliary output `line_logits` with `weighted_bce` at weight `0.10`.
- Harness-owned `bce_dice` primary loss.
- Harness-owned `adamw` optimizer with learning rate `0.001`.
- Harness-owned deterministic shuffle Sampling Policy.
- Harness-owned `temporal_eligible_center` Frame Selection Policy for a matched temporal control.
- No augmentation (`augmentation_policy: none`).
- Harness-owned reduce-on-plateau scheduler.
- Harness-owned early stopping with best-checkpoint restoration.

No custom data loading, custom frame filtering, custom training loop, custom loss, filesystem access, network access, pretrained weights, or artifact writing is introduced.

## Budget Requested

One standard GVCCS Working Validation Run with batch size 8 and up to 80 epochs, matching the temporal ASPP-context and single-frame ASPP-context training budgets. The model architecture is unchanged from the successful single-frame ASPP-context Run and should remain below the current 10M-parameter smoke-test budget.

## Success Criteria

Interpretive success: the completed Run records the temporal-eligible train/validation sample counts and provides an apples-to-apples single-frame control for `run_20260618_204556_e6a60b`.

Strong evidence for temporal channels: best-validation `val/dice` remains at least `0.003` below the temporal Run's `0.873707` while matching the same frame-selection policy.

Evidence for data-policy effect: best-validation `val/dice` is within `0.002` of the temporal Run or substantially above the all-target-frame single-frame ASPP-context Run's `0.866737` without temporal input.

Failure/regression: Run rejection because the configured Research Problem Spec does not allow `temporal_eligible_center` for `single_frame_rgb`, best-validation `val/dice` below `0.8660`, or resource/training instability despite using a previously successful architecture.

## Fallback/Next Decision

If the contract rejects this matched control, create a Capability Request for a Harness-owned single-frame temporal-eligible frame-selection control. If the control nearly matches the temporal Run, pause temporal architecture elaboration and study frame-selection/data-policy effects. If the temporal Run remains clearly ahead, continue temporal-family exploration, prioritizing small precision/recall refinements or bounded diagnostics rather than another uncontrolled data-policy change.
