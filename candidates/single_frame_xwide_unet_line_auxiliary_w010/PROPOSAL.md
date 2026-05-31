# Experiment Proposal: Extra-wide Line Target auxiliary U-Net

## Hypothesis

Increasing the current best base-48 Line Target auxiliary U-Net to a base-64 width will improve Contrail Mask Dice on the GVCCS Working Validation Split by adding representational capacity while preserving the architecture family, Line Target guidance, deterministic shuffle sampling, and no-augmentation data policy that produced the current best Result.

## Comparison Target

Primary comparison target: `run_20260530_101019_def893` from Candidate Experiment `single_frame_wide_unet_line_auxiliary_w010`.

Secondary context: `run_20260530_073538_f7c8b7` (`single_frame_large_unet_line_auxiliary_w010`) and the recent failure-bucket evaluation `eval_eval_2026_05_30_wide_unet_failure_buckets`.

## Expected Effect

The base-48 model improved best-validation `val/dice` to about `0.82694` and its failure-bucket evaluation did not reveal a blocking recall collapse. A moderate width increase should improve thin-contrail and ambiguous-background segmentation by giving the shared encoder/decoder more capacity. The expected risk is overfitting or resource pressure; the requested batch size is reduced to keep the Run conservative.

## Implementation Sketch

Implement the same single-frame three-downsample U-Net family as `single_frame_wide_unet_line_auxiliary_w010`, but set `base_channels=64`. Keep separate 1x1 heads for `mask_logits` and `line_logits`. Keep the same manifest-level Line Target auxiliary loss weight (`0.10`), `bce_dice` primary loss, AdamW optimizer, deterministic shuffle sampling, and no augmentation.

## Contract Features Used

- Model Architecture variation in `model.py` only.
- `input_mode: single_frame_rgb`.
- `output_form: mask_logits`.
- Harness-derived `line` Auxiliary Target with `line_logits`, `weighted_bce`, and weight `0.10`.
- Harness-owned `data.sampling_policy: deterministic_shuffle`.
- Harness-owned `data.augmentation_policy: none`.
- Harness-owned training knobs: `adamw`, learning rate `0.001`, batch size `8`, max epochs `30`.

## Budget Requested

One GVCCS Working Validation Split Run with max epochs 30. Request batch size 8 to reduce GPU memory risk for the wider model. No Post-Run Evaluation is requested until the Run result indicates whether whole-validation failure-bucket diagnostics are useful.

## Success Criteria

- Candidate validates and smoke-tests within the existing Candidate Experiment Contract and parameter budget.
- Completed Run exceeds the comparison target's best-validation `val/dice` (`0.82694`) or achieves a near tie with improved precision/recall balance.
- No Resource Failure or contract violation occurs.

## Fallback/Next Decision

If Dice improves, request or run a bounded failure-bucket Post-Run Evaluation before treating the new model as robustly better. If it regresses without failure, treat base-48 as the current capacity sweet spot and pivot to in-contract training-knob or future Harness-owned loss/scheduler capability rather than continuing width scaling. If it fails from resource pressure, classify it as a Resource Failure and consider a smaller architectural refinement rather than another width increase.
