# Post-Run Evaluations are run-scoped Harness operations

Post-Run Evaluation is a first-class Harness operation that reloads a completed Run's copied Candidate Experiment, Resolved Manifest, and persisted model artifact to compute diagnostics without retraining.

Evaluation artifacts are stored under the original Run at `outputs/evaluations/<evaluation_id>/`; current evaluation IDs derive from request IDs such as `eval_<request_id>`. The implementation writes final artifacts such as `summary.json` and `evaluation_metadata.json` and records ledger events; it does not persist a separate `running` / `completed` / `failed` lifecycle file.

Post-Run Evaluation does not create a new Run, Candidate Experiment, or top-level evaluation tree. This keeps provenance attached to the trained model being examined while distinguishing training Results from later diagnostic analysis.
