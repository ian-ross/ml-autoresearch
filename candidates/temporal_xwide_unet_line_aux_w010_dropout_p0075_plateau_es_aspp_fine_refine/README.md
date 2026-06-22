# temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_fine_refine

Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

This Candidate Experiment starts from the centered temporal RGB clip extra-wide U-Net with Line Target auxiliary output, p=0.075 bottleneck dropout, and lightweight dilated ASPP-style bottleneck context. It adds a small high-resolution `FineScaleRefinement` block after the final decoder stage. The refinement uses parallel depthwise-separable 3x3 convolutions at dilation 1 and dilation 2, fuses them with a 1x1 convolution, and adds the result residually before the mask and line heads.

The intent is to improve thin/faint small-positive recovery while avoiding the blunt precision/recall tradeoff seen with a decoder-side mask gate.

## Manifest choices

- `input_mode: centered_temporal_rgb_clip`
- `output_form: mask_logits`
- Auxiliary target: Harness-derived `line` target via `line_logits`, `weighted_bce`, weight `0.10`
- Data policy: `deterministic_shuffle`, `temporal_eligible_center`, no augmentation
- Training: `bce_dice`, `adamw`, learning rate `0.001`, batch size `8`, max epochs `80`
- Scheduler: Harness-owned `reduce_on_plateau` with factor `0.5`, patience `5`, min LR `1e-5`
- Early stopping: enabled, patience `12`, min delta `0.001`, restore best checkpoint

## Contract assumptions

Candidate code is limited to Model Architecture definition in `model.py`. It does not implement data loading, target derivation, losses, training loops, filesystem access, network access, artifact writing, or pretrained weight handling.

## Known limitations

The primary comparison is against the temporal ASPP-context parent on the temporal-eligible subset, not against all-target-frame single-frame Runs. If the refinement improves aggregate metrics, a bounded failure-bucket Post-Run Evaluation is needed to verify missed-small-positive improvement without increased empty-mask false positives.
