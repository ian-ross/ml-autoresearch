# Evaluation Request format

Evaluation Requests authorize bounded, Harness-owned autonomous Post-Run Evaluations. Autonomous request-gated evaluations use YAML mappings validated before any evaluation artifacts are written.

## Whole-Validation Failure Analysis and request modes

**Whole-Validation Failure Analysis** is the current umbrella diagnostic for evaluating a completed Run across the Working Validation Split without retraining. It reloads the Run's persisted model artifact, computes whole-validation aggregate and per-sample metrics, can compute a probability-threshold sweep, and can save bounded failure-bucket diagnostic samples.

There are two current surfaces for this diagnostic:

- `evaluate-run` is the manual/operator convenience command. It runs the implemented Whole-Validation Failure Analysis directly and creates an implicit JSON Evaluation Request artifact from the CLI/API parameters under `outputs/evaluations/<evaluation_id>/evaluation_request.json` so the durable artifacts have ledger linkage.
- `run-post-run-evaluation` is the autonomous request-gated surface. Its approved `evaluation_mode` values, currently `threshold_sweep` and `failure_bucket_review`, select bounded parts of Whole-Validation Failure Analysis after validating an explicit Evaluation Request. When `[candidate_execution] backend = "docker"`, request-gated Post-Run Evaluation runs inside the configured Candidate Execution Boundary instead of requiring Torch in the Research Workspace host environment.

Required fields:

- `target_run_id`: parent Run to evaluate.
- `evaluation_mode`: approved Harness-owned request mode. Current values: `threshold_sweep`, `failure_bucket_review`. These are not separate ad hoc evaluators; they are bounded request-gated modes within Whole-Validation Failure Analysis.
- `diagnostic_question`: question the diagnostic should answer.
- `expected_decision_impact`: how the result may affect the next Research Loop decision.
- `parameters`: bounded diagnostic parameters, including optional `primary_threshold` in `[0, 1]`, `threshold_sweep` bounds (`min`, `max`, `steps`), `batch_size`, `artifact_count`, and `failure_bucket_count`.
- `artifact_budget`: resource budget with `max_artifacts` and `max_runtime_seconds`.

Implemented bounds include `batch_size <= 1024`, `artifact_count <= 100`, `failure_bucket_count <= 20`, and `max_runtime_seconds <= 3600`.

`request_id` is optional. If omitted, the Harness uses the request filename stem as the stable request identifier.

Run a Post-Run Evaluation with the CLI. In a Research Workspace configured with `candidate_execution.backend = "docker"`, this command dispatches through the configured Docker Candidate Execution Boundary:

```bash
ml-autoresearch run-post-run-evaluation \
  --request evaluation-request.yaml \
  --runs-root runs \
  --ledger-path research-ledger.jsonl
```

The command requires a validated Evaluation Request. Invalid modes or out-of-bounds parameters fail before artifacts or successful evaluation ledger events are created.

Evaluation outputs are written under the parent Run. Every request-gated evaluation writes:

```text
runs/<run_id>/outputs/evaluations/<evaluation_id>/evaluation_metadata.json
runs/<run_id>/outputs/evaluations/<evaluation_id>/summary.json
```

`failure_bucket_review` also runs bounded Whole-Validation Failure Analysis and writes:

```text
runs/<run_id>/outputs/evaluations/<evaluation_id>/aggregate_metrics.json
runs/<run_id>/outputs/evaluations/<evaluation_id>/per_sample_metrics.jsonl
runs/<run_id>/outputs/evaluations/<evaluation_id>/threshold_sweep.json
runs/<run_id>/outputs/evaluations/<evaluation_id>/diagnostic_samples/samples.json
runs/<run_id>/outputs/evaluations/<evaluation_id>/diagnostic_samples/sample_*.png
```

Diagnostic samples are selected deterministically from Harness-owned buckets such
as `worst_by_dice`, `false_positive_heavy`, `false_negative_heavy`,
`empty_mask_false_positives`, and `missed_positive_masks`, within the validated
artifact budget.

`evaluation_metadata.json` links the output to `request_id`, `parent_run_id`, the approved mode, validated parameters, and relative artifact paths.

Research Ledger events:

- `evaluation_requested`: `evaluation_request_id`, `request_path`, `run_id`, `evaluation_mode`.
- `evaluation_completed`: `evaluation_id`, `evaluation_request_id`, `run_id`, `evaluation_mode`, `artifact_metadata_path`.
