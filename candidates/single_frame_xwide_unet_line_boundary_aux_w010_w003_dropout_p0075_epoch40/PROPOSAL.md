# Experiment Proposal: conservative Boundary Target auxiliary head on the 40-epoch p=0.075 extra-wide Line Target U-Net

## Hypothesis

The current best in-contract Result, `run_20260601_162117_66bf89`, improved both precision and recall with a 40-epoch extra-wide base-64 U-Net, Line Target auxiliary weight 0.10, and bottleneck `Dropout2d(p=0.075)`, but its failure-bucket review still showed false-positive spillover on some positive masks and false-negative-heavy under-segmentation on larger masks. A conservative Harness-derived Boundary Target auxiliary head at weight 0.03 should sharpen edge-local features enough to reduce spillover and under-segmentation without overwhelming the successful Line Target supervision.

## Comparison Target

Primary comparison target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`, the current best in-contract 40-epoch p=0.075 extra-wide Line Target U-Net.

Secondary context: `run_20260601_085755_25cd06` for the 30-epoch p=0.075 base and `run_20260601_121820_adbab7` as the earlier boundary-auxiliary runtime-support failure. This proposal is a new Candidate Experiment against the 40-epoch best Result, not a Repair Candidate for the failed 30-epoch boundary attempt.

## Expected Effect

Expected best-validation effect is a small Dice improvement or parity with cleaner precision/recall balance. The boundary auxiliary loss may improve edge localization, reduce broad false-positive spillover, and recover some false-negative-heavy large positives. Because the 40-epoch base is already strong and threshold-stable, success does not require a large Dice gain; maintaining Dice while improving precision or reducing failure-bucket artifacts would still be useful evidence for Boundary Target auxiliary supervision.

## Implementation Sketch

Start from the 40-epoch p=0.075 extra-wide base-64 U-Net architecture. Keep the existing encoder, decoder, bottleneck dropout, mask head, and `line_logits` head. Add only a third 1x1 convolution head named `boundary_logits` from the final decoder feature map. Return exactly `mask_logits`, `line_logits`, and `boundary_logits`. Do not derive boundary targets, implement losses, alter data loading, or add architecture-independent policy code.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Primary mask logits output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`, `weighted_bce`, weight 0.10).
- Harness-derived Boundary Target auxiliary output (`boundary_logits`, `weighted_bce`, weight 0.03).
- Harness-owned `bce_dice` primary loss and AdamW optimizer.
- Deterministic shuffle Sampling Policy and no augmentation.
- Bounded training knobs: learning rate 0.001, batch size 8, max epochs 40.

## Budget Requested

One GVCCS Working Validation Split Run with the same batch size and 40-epoch budget as `run_20260601_162117_66bf89`. Expected model size remains effectively unchanged from the current best plus one 1x1 auxiliary head and stays below the 10M-parameter policy.

## Success Criteria

Primary success: best-validation `val/dice` exceeds `run_20260601_162117_66bf89` by at least +0.001 without reducing either precision or recall by more than 0.005.

Secondary success: if best-validation Dice is within ±0.001 of the comparison target, accept as useful evidence only if precision improves by at least +0.003 or a follow-up failure-bucket evaluation shows lower false-positive-heavy / false-negative-heavy burden without worsening missed positives.

Operational success: the Run validates and trains with both auxiliary targets using Harness-owned target derivation and losses; no Candidate Experiment code may emulate boundary targets or implement custom losses.

## Fallback/Next Decision

If the Run fails with unsupported Boundary Target handling, do not submit another boundary Repair Candidate in this line; write a Research Note or Capability Request depending on whether current docs and runtime are still inconsistent. If it completes but regresses by more than 0.001 Dice or materially worsens precision/recall, reject this conservative boundary weight and return to non-boundary in-contract hypotheses or await stronger Boundary Target support diagnostics. If it matches or improves the comparison target, request a bounded failure-bucket Post-Run Evaluation before further architecture changes.
