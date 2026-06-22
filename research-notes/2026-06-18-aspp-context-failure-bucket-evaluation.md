# 2026-06-18 ASPP-context failure-bucket evaluation

## Hypothesis

The bounded failure-bucket Post-Run Evaluation for `run_20260615_140810_2bee94` tested whether the ASPP-style bottleneck-context extra-wide Line Target U-Net preserved its aggregate Dice/recall gain across the full GVCCS Working Validation Split, or whether the improvement was mainly a precision/recall tradeoff that worsened known failure buckets relative to the plateau/es comparison target `run_20260614_124226_05e3eb`.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context/PROPOSAL.md`
- Primary Comparison Target: `run_20260614_124226_05e3eb` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`

## Run(s)

- Target Run ID: `run_20260615_140810_2bee94`
- Evaluation Request: `evaluation-requests/eval-2026-06-18-aspp-context-failure-buckets.yaml`
- Evaluation ID: `eval_eval_2026_06_18_aspp_context_failure_buckets`
- Evaluation artifact root: `runs/run_20260615_140810_2bee94/outputs/evaluations/eval_eval_2026_06_18_aspp_context_failure_buckets/`
- Comparison evaluation: `eval_eval_2026_06_15_plateau_es_failure_buckets` for `run_20260614_124226_05e3eb`

## Key metrics

Whole-validation aggregate metrics at threshold 0.5:

| Run | Evaluation | Dice | IoU | Precision | Recall | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260614_124226_05e3eb` | `eval_eval_2026_06_15_plateau_es_failure_buckets` | 0.861625 | 0.756890 | 0.883590 | 0.840726 | 3889 |
| `run_20260615_140810_2bee94` | `eval_eval_2026_06_18_aspp_context_failure_buckets` | 0.866738 | 0.764817 | 0.870191 | 0.863312 | 3889 |

The ASPP-context Run preserves the aggregate improvement under whole-validation evaluation: Dice improves by about `+0.00511`, IoU by `+0.00793`, and recall by `+0.02259` over the plateau/es target. Precision drops by about `-0.01340`, confirming that the gain is sensitivity-biased.

Threshold sweep summary:

| Run | Best threshold by Dice | Best sweep Dice | Dice @ 0.25 | Dice @ 0.35 | Dice @ 0.50 | Dice @ 0.60 | Dice @ 0.65 | Dice @ 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260614_124226_05e3eb` | 0.25 | 0.862611 | 0.862611 | 0.862357 | 0.861625 | 0.860712 | 0.860138 | 0.858191 |
| `run_20260615_140810_2bee94` | 0.50 | 0.866738 | 0.865264 | 0.866247 | 0.866738 | 0.866476 | 0.866162 | 0.864555 |

The improvement is not a threshold-local artifact. The ASPP-context Run stays above the plateau/es Run across the inspected thresholds. At threshold 0.65 it recovers precision above the original proposal guardrail (`0.881450`) while retaining Dice `0.866162` and recall `0.851395`, still better than the plateau/es default-threshold aggregate.

Failure-bucket and per-sample summaries at threshold 0.5:

| Quantity | Plateau/es | ASPP-context | Direction |
| --- | ---: | ---: | --- |
| Empty-mask samples | 1004 | 1004 | unchanged |
| Empty masks with any predicted positives | 130 | 128 | slightly improved |
| Mean predicted-positive pixels on empty masks | 0.846 | 0.474 | improved |
| Maximum predicted-positive pixels on an empty mask | 50 | 29 | improved |
| Positive-mask samples | 2885 | 2885 | unchanged |
| Missed positive masks with zero predicted pixels | 73 | 69 | improved, but not solved |
| Mean per-sample false-positive pixels | 55.71 | 64.77 | worse |
| Mean per-sample false-negative pixels | 80.10 | 68.74 | improved |
| Mean false-positive pixels on positive-mask samples | 74.80 | 87.14 | worse |
| Mean false-negative pixels on positive-mask samples | 107.98 | 92.67 | improved |

## Qualitative observations

The diagnostics support a mixed but useful outcome. The ASPP-context model improves the target missed-positive problem: fully missed positive masks fall from 73 to 69, and mean false negatives fall substantially. It does not create broad empty-sky spread; empty-mask affected count, mean empty-mask predicted pixels, and worst empty-mask tail all improve relative to plateau/es.

The cost is concentrated in positive-mask false positives rather than empty-mask false positives. Mean false positives on positive-mask samples increase from 74.80 to 87.14, explaining the aggregate precision drop. This suggests the context block expands detections around real contrail cases rather than hallucinating many empty-sky positives. That tradeoff may be acceptable as a new family baseline, but it argues against simply increasing context capacity further.

Selected diagnostic artifacts:

![ASPP-context missed positive overlay](../runs/run_20260615_140810_2bee94/outputs/evaluations/eval_eval_2026_06_18_aspp_context_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![ASPP-context empty-mask false-positive overlay](../runs/run_20260615_140810_2bee94/outputs/evaluations/eval_eval_2026_06_18_aspp_context_failure_buckets/diagnostic_samples/sample_005_overlay.png)

![ASPP-context false-positive-heavy overlay](../runs/run_20260615_140810_2bee94/outputs/evaluations/eval_eval_2026_06_18_aspp_context_failure_buckets/diagnostic_samples/sample_009_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-aspp-context-missed-positive-overlay
    source_evaluation_id: eval_eval_2026_06_18_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Worst-by-Dice missed-positive diagnostic for the ASPP-context Run; selected to show that small positive masks are still fully missed despite the reduced missed-positive count.
  - figure_id: fig-aspp-context-empty-mask-fp-overlay
    source_evaluation_id: eval_eval_2026_06_18_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_005_overlay.png
    reason: Empty-mask false-positive diagnostic; selected because empty-mask false positives improved rather than explaining the precision drop.
  - figure_id: fig-aspp-context-false-positive-heavy-overlay
    source_evaluation_id: eval_eval_2026_06_18_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_009_overlay.png
    reason: False-positive-heavy diagnostic; selected because the main regression is increased false-positive pixels on positive-mask samples, not broad empty-sky spread.
```

## Decision

Promote `run_20260615_140810_2bee94` as the current best in-contract Comparison Target for architecture work, with an explicit caveat that its improvement is recall-biased and increases false positives on positive-mask samples. The Post-Run Evaluation answers the main concern from the aggregate note: the precision drop is not primarily caused by broad empty-mask hallucination, and the Run remains better than plateau/es across the threshold sweep.

## Next proposed change

Use the ASPP-context Run as the next comparison target, but do not increase context width/depth. The next Candidate Experiment should try to recover precision while preserving the recall gain, for example by adding a lightweight decoder-side gating/refinement mechanism or reducing context-channel expansion within the existing Candidate Experiment Contract. If subsequent variants cannot recover precision, a Harness-owned data-policy or loss-capability direction may be more useful than more architecture-side context.
