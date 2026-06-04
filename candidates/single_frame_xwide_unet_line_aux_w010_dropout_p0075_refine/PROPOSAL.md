# Experiment Proposal: high-resolution refinement for p=0.075 extra-wide line-auxiliary U-Net

## Hypothesis

The current best in-contract Result, `run_20260601_085755_25cd06`, improved Dice and recall with `Dropout2d(p=0.075)` but still misses tiny positive masks and under-segments some larger positives. Adding a small high-resolution residual refinement block after the final U-Net decoder should preserve the successful base architecture while giving the mask and line heads a final local/mid-scale feature mixing stage at full resolution. This may recover small contrail fragments and sharpen thin structures without requiring Boundary Target support or any custom Harness behavior.

## Comparison Target

Primary comparison target: `run_20260601_085755_25cd06` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`, the current best extra-wide single-frame U-Net with Line Target auxiliary weight 0.10 and bottleneck dropout p=0.075.

Global-best comparison is the same Run at proposal time (`val/dice` 0.8346029408714811 best-validation Dice).

## Expected Effect

Expected improvements are modest but targeted:

- higher recall on tiny positive masks by retaining and remixing full-resolution decoder features before the heads;
- equal or slightly better best-validation Dice than the p=0.075 base;
- no material increase in empty-mask false positives compared with the p=0.075 failure-bucket evaluation profile.

A regression would indicate that the remaining failures are less architecture-refinement limited and more likely require approved target/loss/data-policy capability slices.

## Implementation Sketch

Start from the p=0.075 extra-wide base-64 U-Net with the existing Harness-derived `line_logits` auxiliary head. Insert a lightweight `RefinementBlock` after the final decoder stage:

- project the 64-channel final feature tensor through parallel full-resolution 3x3 convolutions with dilation 1 and dilation 2;
- concatenate the two branches, fuse them with a 1x1 convolution, BatchNorm, and ReLU;
- add the fused features residually to the original final decoder tensor;
- feed both `mask_head` and `line_head` from the refined features.

The Candidate Experiment code remains model-architecture only and does not implement data loading, target derivation, loss computation, post-processing, file access, or training control.

## Contract Features Used

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Auxiliary Target: `line` via `line_logits`, `weighted_bce`, weight `0.10`
- Primary loss: Harness-owned `bce_dice`
- Optimizer: Harness-owned `adamw`
- Training knobs: learning rate `0.001`, batch size `8`, max epochs `30`
- Data policy: `deterministic_shuffle`, augmentation policy `none`

No Boundary Target, custom losses, custom augmentations, custom samplers, pretrained weights, runtime downloads, or filesystem/network access are requested.

## Budget Requested

One standard GVCCS Working Validation Split Run with the same budget as the comparison target: batch size 8 and up to 30 epochs. Parameter count should remain under the current 10M smoke-test budget because the refinement block adds only a small number of convolution parameters to the existing extra-wide U-Net.

## Success Criteria

Treat the experiment as promising if it meets both of these criteria:

- best-validation `val/dice` is at least equal to the comparison target within noise and preferably exceeds `0.8346`;
- recall improves or remains close without a large precision/empty-mask false-positive penalty in subsequent failure-bucket diagnostics.

A clear Dice regression below the p=0.10 dropout model (`run_20260530_180658_0af8a8`) or obvious false-positive expansion should reject this refinement direction.

## Fallback/Next Decision

If the Run improves or matches the current best, request or perform a bounded failure-bucket review focused on tiny missed positives, large-mask under-segmentation, and empty-mask false positives. If it regresses, abandon this refinement block and return to either another safe in-contract architectural hypothesis or wait for the pending Boundary Target runtime Capability Request to be resolved. Do not emulate Boundary Target behavior inside candidate code.
