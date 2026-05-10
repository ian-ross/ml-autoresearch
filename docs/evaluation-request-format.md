# Evaluation Request format

Evaluation Requests authorize bounded, Harness-owned autonomous Post-Run Evaluations. They are YAML mappings validated before any evaluation artifacts are written.

Required fields:

- `target_run_id`: parent Run to evaluate.
- `evaluation_mode`: approved Harness-owned mode. Current values: `threshold_sweep`, `failure_bucket_review`.
- `diagnostic_question`: question the diagnostic should answer.
- `expected_decision_impact`: how the result may affect the next Research Loop decision.
- `parameters`: bounded diagnostic parameters, including optional `primary_threshold` in `[0, 1]`, `threshold_sweep` bounds (`min`, `max`, `steps`), `batch_size`, `artifact_count`, and `failure_bucket_count`.
- `artifact_budget`: resource budget with `max_artifacts` and `max_runtime_seconds`.

`request_id` is optional. If omitted, the Harness uses the request filename stem as the stable request identifier.

Run a Post-Run Evaluation with:

```bash
ml-autoresearch run-post-run-evaluation \
  --request evaluation-request.yaml \
  --runs-root runs \
  --ledger-path research-ledger.jsonl
```

The command requires a validated Evaluation Request. Invalid modes or out-of-bounds parameters fail before artifacts or successful evaluation ledger events are created.

Evaluation outputs are written under the parent Run:

```text
runs/<run_id>/evaluations/<evaluation_id>/evaluation_metadata.json
runs/<run_id>/evaluations/<evaluation_id>/summary.json
```

`evaluation_metadata.json` links the output to `request_id`, `parent_run_id`, the approved mode, validated parameters, and relative artifact paths.

Research Ledger events:

- `evaluation_requested`: `evaluation_request_id`, `request_path`, `run_id`, `evaluation_mode`.
- `evaluation_completed`: `evaluation_id`, `evaluation_request_id`, `run_id`, `evaluation_mode`, `artifact_metadata_path`.
