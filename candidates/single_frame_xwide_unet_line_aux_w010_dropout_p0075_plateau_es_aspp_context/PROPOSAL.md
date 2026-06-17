# Experiment Proposal: plateau/es xwide U-Net with lightweight dilated bottleneck context

## Hypothesis

Adding a lightweight dilated multi-scale context block at the extra-wide U-Net bottleneck will reduce missed small/faint positive masks for Ground-Camera Contrail Detection while preserving the precision gains from `run_20260614_124226_05e3eb`. The plateau/es failure-bucket evaluation showed robust aggregate improvement but nearly unchanged fully missed positives (73 vs 74 for the fixed-60 parent). A bottleneck context module should help distinguish faint linear contrails from local cloud texture without reintroducing the high-resolution detail-fusion path that previously reduced precision.

## Comparison Target

Primary Comparison Target: `run_20260614_124226_05e3eb` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`.

Secondary context: `run_20260603_094446_05dea3` showed that direct high-resolution detail fusion raised recall but lost too much precision. This proposal intentionally avoids extra full-resolution encoder-detail fusion and changes only bottleneck context.

## Expected Effect

Expected outcome is a small increase in best-validation Dice and recall, especially fewer fully missed positive masks if a later failure-bucket evaluation is requested, with precision no worse than a small tolerance. The desired direction is improved sensitivity to faint/small contrails without increasing broad empty-sky false positives.

## Implementation Sketch

Start from the current-best extra-wide base-64 U-Net with p=0.075 bottleneck dropout and Line Target auxiliary output. Insert a `DilatedContextBlock` after bottleneck dropout and before the first decoder upsample. The block uses parallel bottleneck-space convolutions with dilation rates 1, 2, 4, and 6, each projecting 512 channels to 96 channels, concatenates their outputs, and fuses them back to 512 channels with a 1x1 projection, batch normalization, and ReLU. Prediction heads remain unchanged and are attached to the final decoder features.

## Contract Features Used

- `single_frame_rgb` Input Mode.
- `mask_logits` primary output.
- Harness-derived Line Target auxiliary output `line_logits` with `weighted_bce` at weight `0.10`.
- Harness-owned `bce_dice` primary loss.
- Harness-owned `adamw` optimizer with learning rate `0.001`.
- Harness-owned deterministic shuffle Sampling Policy.
- No augmentation (`augmentation_policy: none`).
- Harness-owned reduce-on-plateau scheduler.
- Harness-owned early stopping with best-checkpoint restoration.

No custom data loading, custom training loop, custom loss, filesystem access, network access, pretrained weights, or artifact writing is introduced.

## Budget Requested

One standard GVCCS Working Validation Run with batch size 8 and up to 80 epochs, matching the plateau/es comparison target. The architecture is intended to remain below the current 10M-parameter smoke-test budget; the added context block uses compact 96-channel branches.

## Success Criteria

Strong success: best-validation `val/dice` exceeds `0.8640` and `val/recall` improves over `0.8407` while `val/precision` remains at least `0.8780`.

Minimal success: best-validation `val/dice` matches or exceeds the comparison target (`0.8616`) with either improved recall or qualitative evidence of fewer missed small positives, without more than about `0.005` precision loss.

Failure/regression: best-validation `val/dice` below `0.8610`, precision below `0.8780`, a resource failure from exceeding parameter/memory limits, or signs of late-epoch instability not mitigated by best-checkpoint restoration.

## Fallback/Next Decision

If this candidate improves aggregate metrics, request a bounded failure-bucket Post-Run Evaluation focused on missed positive masks, false-positive-heavy positives, and empty-mask false positives before promoting it. If it regresses through precision loss, abandon bottleneck context for now and prefer either a Harness-owned data-policy capability for positive/negative balancing or a smaller precision-safe training-policy variant. If it fails due to resource limits, submit at most one repair with smaller context branch width while preserving this hypothesis and Comparison Target.
