# 2026-06-18 ASPP mask-gate precision tradeoff

## Hypothesis

`single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_mask_gate` tested whether adding a lightweight decoder-side spatial gate to the current best ASPP-context extra-wide Line Target U-Net would recover precision lost by `run_20260615_140810_2bee94` while preserving most of the ASPP-context recall and Dice gain over the plateau/es parent.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_mask_gate`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_mask_gate`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_mask_gate/PROPOSAL.md`
- Primary Comparison Target: `run_20260615_140810_2bee94` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Secondary reference: `run_20260614_124226_05e3eb` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`

## Run(s)

- Run ID: `run_20260618_113001_287165`
- Dataset mode/subset: GVCCS Working Validation Split, 15,558 train samples and 3,889 validation samples.
- Harness/backend notes: completed on CUDA with Docker backend. The model summary reports 9,369,715 parameters, below the 10,000,000 parameter budget and only slightly above the ASPP-context target. Reduce-on-plateau scheduling and early stopping with best-checkpoint restoration were enabled; early stopping fired after 58 completed epochs and restored the best checkpoint from epoch 56.

## Key metrics

Best-validation metrics selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260614_124226_05e3eb` | plateau/es parent | 71 | 0.861622 | 0.756886 | 0.883582 | 0.840728 | 0.484870 | 0.495275 |
| `run_20260615_140810_2bee94` | ASPP-context target | 63 | 0.866737 | 0.764816 | 0.870191 | 0.863310 | 0.470763 | 0.479140 |
| `run_20260618_113001_287165` | ASPP mask gate | 56 | 0.863796 | 0.760247 | 0.877939 | 0.850101 | 0.472557 | 0.481186 |

The mask gate recovered about `+0.00775` precision over the ASPP-context target, but lost about `-0.01321` recall and `-0.00294` Dice. Relative to the plateau/es parent, it still improves Dice by about `+0.00217`, IoU by `+0.00336`, and recall by `+0.00937`, but precision remains lower by about `-0.00564`.

The proposal's strong success criterion was not met because Dice did not match or exceed `0.8667`, even though precision improved and recall remained above `0.8550` only failed by about `0.0049`. The minimal success criterion was also narrowly missed: best `val/dice` was `0.863796` versus the `0.8640` threshold, with precision improved and recall still above the plateau/es target.

Final completed-epoch metrics for `run_20260618_113001_287165` were close to the best checkpoint: final `val/dice` 0.862957, `val/iou` 0.758948, `val/precision` 0.879973, `val/recall` 0.846586, `val/loss` 0.474849, and `val/total_loss` 0.483477. The best-to-final Dice gap is only about `0.00084`, much smaller than the ASPP-context target's best-to-final gap.

## Qualitative observations

The two saved first-N prediction samples show the same precision/recall tradeoff in miniature rather than a uniform improvement. Compared with the ASPP-context target, sample `val/000000` improves Dice from 0.890511 to 0.904110, but sample `val/000001` falls from 0.943662 to 0.888889. This is consistent with a gate that can suppress some over-expansion but sometimes trims useful positive structure.

![Mask-gate sample 000 overlay](../runs/run_20260618_113001_287165/outputs/prediction_samples/sample_000_overlay.png)

![Mask-gate sample 001 overlay](../runs/run_20260618_113001_287165/outputs/prediction_samples/sample_001_overlay.png)

![Mask-gate sample 001 heatmap](../runs/run_20260618_113001_287165/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-mask-gate-sample-000-overlay
    source_run_id: run_20260618_113001_287165
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the mask-gate Run; selected because this sample improved over the ASPP-context target despite aggregate Dice regression.
  - figure_id: fig-mask-gate-sample-001-overlay
    source_run_id: run_20260618_113001_287165
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the mask-gate Run; selected because its first-N Dice regressed sharply relative to the ASPP-context target, illustrating the recall cost.
  - figure_id: fig-mask-gate-sample-001-heatmap
    source_run_id: run_20260618_113001_287165
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the regressed first-N sample, selected to inspect whether the gate suppresses confidence on true positive structure.
```

## Decision

Do not promote `run_20260618_113001_287165` over the ASPP-context target. The mask gate is scientifically informative because it moves the precision/recall tradeoff in the intended direction, but the aggregate Dice loss means the ungated ASPP-context `run_20260615_140810_2bee94` remains the better in-contract Comparison Target.

This is not a candidate bug or resource failure; it is a completed but mildly regressed research result. The result suggests that decoder-side gating is a plausible precision lever, but this specific residual spatial gate is too blunt or under-supervised to preserve the ASPP-context recall gain.

## Next proposed change

Before stacking more architecture-side refinements, request a bounded failure-bucket Post-Run Evaluation for `run_20260618_113001_287165` only if the next step needs to know whether the precision recovery comes specifically from fewer false-positive-heavy positive samples. Otherwise, keep `run_20260615_140810_2bee94` as the primary target and test a smaller context-capacity variant or a Harness-owned loss/data-policy direction rather than promoting this gate.
