# 2026-05-30 medium U-Net Line Target auxiliary w0.05 regression

## Hypothesis

The `single_frame_medium_unet_line_auxiliary_w005` Candidate Experiment tested whether reducing the Harness-derived Line Target auxiliary loss weight from 0.10 to 0.05 would preserve the medium U-Net's thin-structure bias while recovering recall. The primary Comparison Target was `run_20260529_155844_d8ebec` (`single_frame_medium_unet_line_auxiliary_w010`), whose best-validation `val/dice` was 0.7994525273696043 with precision 0.8429952005805884 and recall 0.760187086855412.

## Candidate Experiment(s)

- Candidate Experiment path or ID: `candidates/single_frame_medium_unet_line_auxiliary_w005`
- Relevant Experiment Proposal: `candidates/single_frame_medium_unet_line_auxiliary_w005/PROPOSAL.md`
- Primary Comparison Target: `run_20260529_155844_d8ebec` / `single_frame_medium_unet_line_auxiliary_w010`
- Secondary context: `run_20260510_165335_b458c3` / `single_frame_small_unet_line_auxiliary_w010`, best-validation `val/dice` 0.7843769771696992 and recall 0.7973861422181288.

## Run(s)

- Run ID: `run_20260529_210959_8b9a18`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed Docker/GPU Run using `input_mode: single_frame_rgb`, primary `bce_dice` loss, Line Target auxiliary `weighted_bce` at weight 0.05, deterministic shuffle, and no augmentation. Batch size requested/effective: 16; max epochs: 30; no resource retry beyond the successful first attempt. The model summary reports 1,096,634 parameters, matching the w0.10 medium U-Net family and staying within the 10,000,000 parameter budget.

## Key metrics

| Run | Candidate Experiment | Epoch source | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260510_165335_b458c3` | `single_frame_small_unet_line_auxiliary_w010` | best epoch 30 | 0.7843769771696992 | 0.6452468918723383 | 0.7717854808163968 | 0.7973861422181288 | 0.5639248168630936 |
| `run_20260529_155844_d8ebec` | `single_frame_medium_unet_line_auxiliary_w010` | best epoch 28 | 0.7994525273696043 | 0.665906634760569 | 0.8429952005805884 | 0.760187086855412 | 0.5434756466587701 |
| `run_20260529_210959_8b9a18` | `single_frame_medium_unet_line_auxiliary_w005` | best epoch 28 | 0.7945989165999967 | 0.659198774202797 | 0.8317512055349874 | 0.7606237182899099 | 0.5469463209040399 |
| `run_20260529_210959_8b9a18` | `single_frame_medium_unet_line_auxiliary_w005` | final epoch 30 | 0.7754407415985798 | 0.6332406833548221 | 0.8192851374676644 | 0.7360506737877605 | 0.576760943826718 |

The w0.05 follow-up did not meet the proposal's success criteria. Best-validation Dice fell by about 0.00485 relative to the w0.10 medium U-Net, outside the allowed match-within-0.003 band. Recall was effectively unchanged at the best epoch (`0.76062` versus `0.76019`), while precision dropped (`0.83175` versus `0.84299`). The final epoch again regressed below the best epoch, reinforcing that this family should be interpreted using persisted best-validation metrics rather than final epoch metrics.

## Qualitative observations

The two saved first-N validation examples are weaker than the w0.10 medium U-Net samples reported in the previous Research Note:

- `val/000000`: Dice 0.6780, IoU 0.5128 for w0.05 versus 0.8919 Dice for the w0.10 medium Run.
- `val/000001`: Dice 0.7075, IoU 0.5474 for w0.05 versus 0.8189 Dice for the w0.10 medium Run.

Because these are only first-N examples, they should not override the aggregate metrics. They are nevertheless consistent with the aggregate regression: lowering the Line Target auxiliary weight did not visibly recover missed structure in these samples and instead produced worse saved overlays.

Example first-N overlays and probability heatmap:

![Sample 0 overlay](../runs/run_20260529_210959_8b9a18/outputs/prediction_samples/sample_000_overlay.png)

![Sample 0 probability heatmap](../runs/run_20260529_210959_8b9a18/outputs/prediction_samples/sample_000_probability_heatmap.png)

![Sample 1 overlay](../runs/run_20260529_210959_8b9a18/outputs/prediction_samples/sample_001_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-medium-unet-w005-sample-000-overlay
    source_run_id: run_20260529_210959_8b9a18
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows the first saved validation prediction for the w0.05 medium U-Net, illustrating the qualitative drop relative to the previous w0.10 medium-family note.
  - figure_id: fig-medium-unet-w005-sample-000-heatmap
    source_run_id: run_20260529_210959_8b9a18
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Provides probability-level context for the weaker first saved prediction after lowering the Line Target auxiliary weight.
  - figure_id: fig-medium-unet-w005-sample-001-overlay
    source_run_id: run_20260529_210959_8b9a18
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Shows a second saved validation prediction where the w0.05 setting remains below the prior w0.10 medium-family qualitative sample.
```

## Decision

Do not promote `run_20260529_210959_8b9a18` to current best. Treat `run_20260529_155844_d8ebec` (`single_frame_medium_unet_line_auxiliary_w010`) as the stronger medium-family baseline by best-validation Dice. The w0.05 result suggests that reducing Line Target auxiliary weight is not a useful recall-recovery lever for this architecture under the current training policy.

## Next proposed change

Stop immediate auxiliary-weight reductions for the medium U-Net family. The more useful next step is to use the already completed failure-bucket Post-Run Evaluation for `run_20260529_155844_d8ebec` to decide whether the w0.10 model's lower recall causes important missed-positive or thin-contrail failures. If that diagnostic shows acceptable failure modes, continue from the w0.10 medium baseline; if it shows severe recall-specific misses, prefer a different in-contract lever rather than further lowering the auxiliary weight.
