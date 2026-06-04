# 2026-06-01 boundary auxiliary training contract failure

## Hypothesis

The failed Candidate Experiment `single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075` was intended to test whether adding a conservative Harness-derived Boundary Target auxiliary head (`boundary_logits`, `weighted_bce`, weight `0.05`) to the current best p=0.075 extra-wide Line Target U-Net would reduce boundary under-segmentation and false-positive spillover while preserving the recall gain from `run_20260601_085755_25cd06`.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075/PROPOSAL.md`
- Primary Comparison Target: `run_20260601_085755_25cd06` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`
- Related Capability Request already pending: `capability-requests/capability-2026-06-01-boundary-target-agent-validator-sync.yaml`

## Run(s)

- Run ID: `run_20260601_121820_adbab7`
- Dataset mode/subset: GVCCS Working Validation Split training path, Docker backend.
- Harness/backend notes: the candidate passed submission and Run start, but failed immediately in Harness-owned training before producing metrics or prediction samples.
- Harness-recorded failure classification: `candidate_bug`.

## Key metrics

No validation metrics were produced. The Run failed before training could compute `outputs/final_metrics.json`, `outputs/best_metrics.json`, or prediction-sample artifacts.

| Run | Candidate Experiment | Status | Failure classification | Metrics |
| --- | --- | --- | --- | --- |
| `run_20260601_121820_adbab7` | `single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075` | failed | `candidate_bug` | none |

The failure reason in `run_metadata.json` is:

```text
ml_autoresearch.errors.TrainingError: unsupported auxiliary target in resolved manifest: {'name': 'boundary', 'output': 'boundary_logits', 'loss': 'weighted_bce', 'weight': 0.05}
```

The resolved manifest preserved the requested auxiliary target block:

```yaml
auxiliary_targets:
- name: line
  output: line_logits
  loss: weighted_bce
  weight: 0.1
- name: boundary
  output: boundary_logits
  loss: weighted_bce
  weight: 0.05
```

## Qualitative observations

There are no model-quality artifacts to inspect. The failure is operational/contract-surface evidence rather than evidence about the scientific value of Boundary Target auxiliary supervision. The static submission path and ingestion accepted the boundary auxiliary manifest in this Autonomy Step, but the GVCCS training path still rejected `boundary` as an unsupported auxiliary target at loss-construction time.

This extends the earlier boundary-target blocker: the mismatch is not only agent-side static validation/documentation synchronization. At least one runtime training component also lacks support for the documented `boundary` auxiliary target surface.

## Research Figures

No Research Figures are included because the Run failed before generating prediction samples or Post-Run Evaluation artifacts.

## Decision

Do not submit a Repair Candidate in this line. Removing `boundary_logits` or silently folding boundary behavior into candidate architecture would change the hypothesis or create a covert workaround; it would no longer test Harness-derived Boundary Target auxiliary supervision. The Harness-recorded `candidate_bug` classification is useful for lifecycle accounting, but the correct research decision is to treat the result as a contract-surface/runtime support blocker.

Keep `run_20260601_085755_25cd06` as the current best in-contract Result. Boundary Target auxiliary training remains blocked until the pending Capability Request is resolved across both static validation/ingestion and runtime training loss support.

## Next proposed change

Wait for human-supervised Harness resolution of `capability_2026_06_01_boundary_target_agent_validator_sync`, expanded if necessary to include runtime training support for `boundary` auxiliary targets. If Boundary Target remains unavailable, the next autonomous Candidate Experiment should use only currently executable in-contract surfaces and should address the p=0.075 model's tiny missed positives or calibration failures without custom target derivation, custom losses, or hidden data-policy changes.
