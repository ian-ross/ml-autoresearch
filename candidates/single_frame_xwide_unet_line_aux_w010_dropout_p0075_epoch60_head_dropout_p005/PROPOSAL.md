# Experiment Proposal: xwide p=0.075 epoch60 with light head-feature dropout

## Hypothesis

The current best in-contract Result, `run_20260602_203450_c05550`, improves aggregate Dice with an extra-wide base-64 U-Net, Line Target auxiliary weight 0.10, bottleneck `Dropout2d(p=0.075)`, deterministic shuffle, no augmentation, and a 60-epoch budget. Its failure-bucket evaluation still shows tiny false positives spread across more empty masks and persistent broad spillover on some positive masks. A light `Dropout2d(p=0.05)` applied only to the final decoder feature map before the mask and line heads may discourage brittle high-resolution texture cues and improve precision/empty-mask cleanliness while preserving the proven encoder/decoder capacity, Line Target supervision, and 60-epoch training behavior.

## Comparison Target

Primary Comparison Target: `run_20260602_203450_c05550` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`.

Global-best context is the same Run. Secondary diagnostic context should include `run_20260603_094446_05dea3`, because that detail-fusion architecture raised recall but lost precision; this proposal intentionally regularizes the final head instead of adding new high-resolution detail capacity.

## Expected Effect

Expected aggregate effect is neutral-to-slightly-positive best-validation Dice with a precision-safe tradeoff: lower or comparable empty-mask false-positive pixels and fewer empty-mask samples with any false positive, without materially increasing missed-positive samples. The head dropout may reduce final-head co-adaptation and broad spillover, but it could also suppress faint positives; the Run should be rejected if the Dice gain is bought by lower recall or more missed-positive masks.

## Implementation Sketch

Start from the current best extra-wide U-Net architecture. Keep the base-64 encoder/decoder, Line Target auxiliary output, bottleneck `Dropout2d(p=0.075)`, mask head, and line head. Add a single `nn.Dropout2d(p=0.05)` after the final `up1` decoder block and before both 1x1 output heads. Return exactly `mask_logits` and `line_logits`.

No data loading, target derivation, loss computation, artifact writing, filesystem access, network access, or training-loop logic is implemented in candidate code.

## Contract Features Used

- Single-Frame RGB Input: `input_mode: single_frame_rgb`
- Primary output: `output_form: mask_logits`
- Harness-derived Line Target auxiliary output: `line_logits`, `weighted_bce`, weight `0.10`
- Harness-owned primary loss: `bce_dice`
- Harness-owned optimizer: `adamw`, learning rate `0.001`
- Harness-owned training knobs: batch size `8`, max epochs `60`
- Harness-owned Data Policy: `deterministic_shuffle` sampling, `none` augmentation
- Candidate code is model architecture only; no custom losses, samplers, transforms, training loops, data loaders, filesystem probes, network calls, or pretrained weights.

## Budget Requested

One GVCCS Working Validation Split Run with CUDA/Docker backend under the standard candidate budget. Requested training knobs are batch size `8` and max epochs `60`. Parameter count should remain essentially unchanged from the comparison target and below the 10M parameter budget because dropout adds no trainable parameters.

## Success Criteria

Promote only if best-validation `val/dice` is at least comparable to the Comparison Target (no regression larger than about `0.002`) and the Result improves precision-oriented behavior: ideally higher `val/precision` or fewer empty-mask false-positive pixels / affected empty masks in follow-up failure-bucket evaluation, while keeping `val/recall` within about `0.005` of `run_20260602_203450_c05550`. The best-to-final Dice gap should remain no worse than about `0.005` so the architecture is not less stable than the current best.

## Fallback/Next Decision

If the Run regresses best-validation Dice, materially lowers recall, or increases the best-to-final Dice gap, treat head-feature dropout as a bad research result and keep `run_20260602_203450_c05550` as the baseline. If the Run is comparable by Dice but precision/empty-mask behavior is ambiguous, request a bounded failure-bucket Post-Run Evaluation before promotion. If it improves Dice and precision-safety, promote it and use it as the next baseline for small in-contract regularization or threshold-insensitive architecture changes.
