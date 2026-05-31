# single_frame_xwide_unet_line_aux_w010_attention_dropout

This Candidate Experiment is an extra-wide base-64 single-frame U-Net for Ground-Camera Contrail Detection. It keeps the current best in-contract settings: mask logits plus Harness-derived Line Target auxiliary logits with `weighted_bce` weight 0.10, deterministic shuffle sampling, no augmentation, AdamW, learning rate 0.001, batch size 8, and 30 maximum epochs.

The architectural change relative to `single_frame_xwide_unet_line_aux_w010_dropout` is lightweight attention gating on all decoder skip connections. Each gate uses the upsampled decoder tensor to generate a one-channel sigmoid mask over the corresponding encoder skip feature map before concatenation. The goal is to reduce irrelevant background detail passed through skips while preserving thin contrail structure.

The model code is architecture-only. It does not implement data loading, training, losses, augmentation, filesystem access, network access, subprocesses, checkpoint downloads, or artifact writing.
