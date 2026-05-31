# Experiment Proposal: Medium U-Net w0.10 with light photometric augmentation

## Hypothesis

The current best in-contract Result, `run_20260529_155844_d8ebec`, came from increasing the Line Target auxiliary U-Net family from base 16 to base 24 while keeping the auxiliary weight at 0.10 and no augmentation. A prior small-U-Net control showed that `light_photometric` augmentation was near-neutral and much less harmful than the geometric-containing presets. This Candidate Experiment tests whether applying the same Harness-owned `light_photometric` augmentation to the stronger medium model improves robustness enough to lift best-validation Dice without the severe regression seen from geometric augmentation.

## Comparison Target

Primary Comparison Target: `run_20260529_155844_d8ebec` / `single_frame_medium_unet_line_auxiliary_w010`, best-validation `val/dice` 0.7994525273696043.

Secondary context: `run_20260525_202245_c3bcd5` showed that `light_photometric` augmentation nearly matched the small unaugmented line-auxiliary baseline while geometric and combined presets regressed substantially.

## Expected Effect

Light photometric augmentation should expose the medium model to small brightness, contrast, and noise perturbations while preserving image/mask geometry. Expected improvement is a small increase in validation robustness and best-validation `val/dice`, or at minimum a match to the unaugmented medium baseline within normal variation. Because the current best medium model is already high precision and lower recall than the small baseline, this proposal does not assume photometric augmentation will recover recall; it asks whether the capacity-improved architecture can absorb photometric regularization without losing its aggregate Dice advantage.

## Implementation Sketch

Start from `single_frame_medium_unet_line_auxiliary_w010`: a base-24 U-Net with shared encoder/decoder features and separate `mask_logits` and `line_logits` heads. Keep the architecture, optimizer, learning rate, batch size, max epochs, deterministic shuffle sampling policy, primary `bce_dice` loss, and Line Target auxiliary `weighted_bce` loss at weight 0.10 unchanged. Change only the Harness-owned data policy from `augmentation_policy: none` to `augmentation_policy: light_photometric`.

## Contract Features Used

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Harness-derived Line Target auxiliary output `line_logits`
- Auxiliary loss `weighted_bce` with weight `0.10`
- Primary loss `bce_dice`
- Optimizer `adamw`
- `data.sampling_policy: deterministic_shuffle`
- `data.augmentation_policy: light_photometric`
- Bounded training knobs: learning rate `0.001`, batch size `16`, max epochs `30`

No custom data loader, custom transform, custom training loop, filesystem access, network access, pretrained weight download, or side-channel artifact writing is requested.

## Budget Requested

One standard GVCCS Working Validation Split Run with the same budget as the current medium-family Runs: batch size 16, max 30 epochs, and the existing Harness resource policy. The architecture remains within the previously observed parameter budget for the base-24 medium U-Net family.

## Success Criteria

Promote this Candidate Experiment only if its best-validation `val/dice` exceeds the primary Comparison Target by at least `+0.003`, or if it matches within `0.003` while materially improving a relevant secondary metric such as recall without an offsetting large precision loss. Interpret final-epoch metrics separately from best-validation metrics because recent medium-family Runs have regressed after their best epoch.

## Fallback/Next Decision

If this Candidate Experiment regresses, treat photometric augmentation as not immediately useful for the medium U-Net family and return to the unaugmented `single_frame_medium_unet_line_auxiliary_w010` as the current best baseline. The next in-contract follow-up should then avoid further available augmentation presets and instead consider a different bounded architecture/training knob, or wait for the pending failure-bucket metrics Capability Request if recall-specific diagnostics are needed.
