# Experiment Proposal: temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context

## Hypothesis

Adding Harness-owned centered three-frame temporal RGB clip input to the current best ASPP-context extra-wide Line Target U-Net will preserve the ASPP model's recall gain while recovering precision on positive-mask false positives. Adjacent GVCCS frames should help the same architecture distinguish temporally persistent linear contrails from single-frame cloud-edge texture without adding decoder-side gates that trimmed true positives.

## Comparison Target

Primary Comparison Target: `run_20260615_140810_2bee94` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`, the current best in-contract architecture result by best-validation Dice (`val/dice` 0.866737 at epoch 63).

Secondary references: `run_20260618_113001_287165` for the failed mask-gate precision/recalibration tradeoff, and `run_20260614_124226_05e3eb` for the plateau/es pre-ASPP parent.

## Expected Effect

The temporal clip should give the first encoder block motion/context evidence unavailable to the single-frame model while keeping all later capacity and training policy nearly unchanged. Expected gains are fewer false-positive pixels on positive-mask samples and improved precision relative to the ASPP-context target, with recall remaining closer to the ASPP-context target than to the mask-gated variant. Because `centered_temporal_rgb_clip` uses only temporally eligible center frames, validation sample counts may differ from single-frame all-target-frame Runs; interpret metrics with this data-policy difference explicit.

## Implementation Sketch

Start from the ASPP-context extra-wide U-Net architecture and change only the input mode/first convolution channel count through `input_spec["shape"]`. The Research Problem adapter supplies a channel-stacked 3-frame RGB clip with shape `[9, 128, 128]`; the model treats those channels as a single tensor and still predicts only the center Target Frame Contrail Mask. Keep the same base-64 encoder/decoder, p=0.075 bottleneck dropout, dilated bottleneck context block, mask head, and Line Target auxiliary head.

## Contract Features Used

- `input_mode: centered_temporal_rgb_clip` with Harness-owned clip construction.
- `data.frame_selection_policy: temporal_eligible_center`, required for temporal clips.
- `output_form: mask_logits` for the primary Contrail Mask.
- Harness-derived Line Target auxiliary output `line_logits` with `weighted_bce` at weight `0.10`.
- `data.sampling_policy: deterministic_shuffle` and `data.augmentation_policy: none`.
- Harness-owned `bce_dice` loss, AdamW optimizer, reduce-on-plateau scheduler, and early stopping with best-checkpoint restoration.

No custom data loading, temporal sampling, target construction, loss, scheduler, filesystem access, network access, or runtime weight download is introduced.

## Budget Requested

Request one standard GVCCS Working Validation Split Run with CUDA/Docker backend if available, batch size 8, max epochs 80, reduce-on-plateau scheduler (`factor: 0.5`, `patience: 5`, `min_lr: 1e-5`), and early stopping (`patience: 12`, `min_delta: 0.001`, restore best checkpoint). Parameter count should remain below the 10M smoke-test budget because only the first convolution grows from 3 to 9 input channels.

## Success Criteria

Strong success: best-validation `val/dice` exceeds `0.866737` while `val/precision` is at least `0.875` and `val/recall` remains at least `0.858` on the temporal-eligible validation set.

Minimal useful success: best-validation `val/dice` is within `0.002` of the ASPP-context target and precision improves over the ASPP-context `0.870191` without dropping recall below the mask-gate `0.850101`. If successful or near-miss, request a bounded failure-bucket Post-Run Evaluation before promoting the temporal family because the eligible-frame validation subset differs from all-target-frame single-frame Runs.

## Fallback/Next Decision

If this regresses clearly, do not keep pushing temporal capacity in architecture code; treat the result as evidence that the smaller temporal-eligible training subset or naive channel stacking is not enough. Return to the ASPP-context single-frame target and consider Harness-owned loss/data-policy directions or a matched single-frame `temporal_eligible_center` control before further temporal variants.
