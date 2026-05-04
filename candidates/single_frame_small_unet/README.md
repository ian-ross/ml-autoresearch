# single_frame_small_unet

A small standard U-Net-style Candidate Experiment for the first architecture variation after the tiny fixture baseline.

Intent:

- stay within the current Candidate Experiment Contract;
- use Single-Frame RGB Input;
- emit only `mask_logits`;
- keep `bce_dice` + AdamW from the baseline manifest;
- add enough encoder/decoder capacity and skip connections to make the tiny GVCCS subset a credible end-to-end check.

This is still a small one-epoch Human-Guided Research Iteration candidate, not a final model-quality claim.
