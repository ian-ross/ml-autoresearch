# single_frame_small_unet_realistic_training

A small standard U-Net-style Candidate Experiment for a more realistic Human-Guided Research Iteration on GVCCS.

Intent:

- stay within the current Candidate Experiment Contract;
- use Single-Frame RGB Input;
- emit only `mask_logits`;
- keep `bce_dice` + AdamW from the baseline manifest;
- reuse the base-16 Small U-Net architecture from `single_frame_small_unet`;
- increase the Harness-owned training budget to `batch_size: 16` and `max_epochs: 30`.

The learning rate remains `0.001`, which is a reasonable starting point for AdamW on this compact U-Net. If validation loss is unstable, a later Candidate Experiment should try a smaller learning rate such as `0.0003`.
