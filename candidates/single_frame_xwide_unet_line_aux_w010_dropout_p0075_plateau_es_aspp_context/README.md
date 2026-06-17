# single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context

Extra-wide single-frame U-Net Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

This candidate keeps the current-best base-64 extra-wide U-Net with p=0.075 bottleneck dropout and Line Target auxiliary head, then adds a lightweight dilated context block at the 512-channel bottleneck. The context block runs parallel bottleneck-space convolutions with dilation rates 1, 2, 4, and 6 using compact 96-channel branches, concatenates them, and projects back to 512 channels before decoding.

The purpose is to give the decoder more multi-scale context for faint or small contrails without adding the high-resolution detail-fusion path that previously improved recall but reduced precision.

## Manifest choices

- Input: `single_frame_rgb`
- Output: `mask_logits`
- Auxiliary target: Harness-derived `line` target via `line_logits`, `weighted_bce`, weight `0.10`
- Data policy: deterministic shuffle, no augmentation
- Training: `bce_dice`, AdamW, learning rate `0.001`, batch size `8`, max epochs `80`
- Training policy: reduce-on-plateau scheduler and early stopping with best-checkpoint restoration

## Contract assumptions

The Candidate Experiment code defines only Model Architecture. The Harness owns data loading, target construction, losses, optimizer, scheduler, early stopping, metrics, artifacts, and validation split handling.

## Known limitations

This is not a temporal model and cannot use neighboring frames. The added context may still trade precision for recall; promotion should depend on whole-validation failure-bucket diagnostics if aggregate metrics improve.
