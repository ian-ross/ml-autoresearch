# single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_mask_gate

Extra-wide single-frame U-Net Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

This candidate starts from the current-best ASPP-context base-64 extra-wide U-Net with p=0.075 bottleneck dropout and a Line Target auxiliary head. It keeps the compact dilated bottleneck context block unchanged, then adds a lightweight decoder-side `SpatialMaskGate` before the primary mask head.

The gate is learned from final decoder features using a small 3x3/1x1 convolutional path and sigmoid. It scales only the features used by the mask head (`0.5 + gate` residual-style scaling). The line head remains attached to ungated features so the Harness-derived Line Target can continue supervising thin structure.

The purpose is to reduce the ASPP-context model's false-positive expansion on positive-mask samples while preserving its improved missed-positive/recall behavior.

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

This is still a single-frame model and cannot use neighboring temporal frames. The mask gate may recover precision at the cost of some recall; promotion should depend on aggregate metrics and, if promising, bounded failure-bucket diagnostics.
