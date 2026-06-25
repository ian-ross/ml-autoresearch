# Harness capabilities and ownership

This document describes generic ML Autoresearch Harness responsibilities. Research Problem-specific values such as input modes, losses, optimizers, frame-selection policies, augmentation presets, auxiliary target names, metrics, and adapter behavior come from the configured `ResearchProblemSpec` and trusted provider package.

GVCCS-specific examples live temporarily in `docs/gvccs-features.md`.

## Harness-owned capabilities

The Harness owns:

- Workspace configuration loading from `ml-autoresearch.toml`.
- Candidate source, manifest, proposal, and static contract validation.
- Controlled smoke testing through `build_model(input_spec, output_spec)`.
- Candidate Run creation, status metadata, run-scoped artifact layout, and logs.
- Training/evaluation orchestration through native or Docker backends.
- Docker data/package mounts and runtime image validation for execution/evaluation/autonomy commands.
- Research Ledger event validation and append-only writes.
- Experiment Batch ingestion and execution policy.
- Post-Run Evaluation request validation and run-scoped evaluation artifacts.
- Agent Control Boundary preparation, curated context snapshots, skill installation, and handoff ingestion.

The Harness does not give Candidate Experiment code authority over arbitrary filesystem paths, data loading, Docker, shell commands, network access, runtime weight downloads, custom training loops, or ledger writes.

## Provider-owned capabilities

A trusted Research Problem provider owns domain choices and adapters, including:

- allowed `input_mode` values and input specs;
- allowed `output_form` values and output specs;
- allowed primary losses and optimizers;
- allowed sampling, frame-selection, and augmentation policies;
- primary selection metric and selection mode;
- auxiliary target names, output names, loss names, and target construction;
- dataset discovery/loading and split policy;
- training and evaluation adapter hooks;
- Research Problem briefs and dataset profile artifacts.

Generic docs should not treat a provider example as a universal Harness allowlist. Candidate manifests are valid only when their selected values are advertised by the active Research Problem Spec.

## Execution model

`submit-candidate` performs static validation and submission. It does not require Runtime Image validation.

`run-candidate` validates, smoke-tests, and trains synchronously against the configured Research Problem provider. Docker is the default Candidate Execution Boundary for `run-candidate`; the native backend remains an explicit developer-unsafe escape hatch.

For Docker-backed Research Problem training, the Harness validates configured provider data roots, mounts approved host data read-only at `/data`, mounts the trusted Research Problem package read-only, and writes outputs under the run's `outputs/` directory.

## Candidate model interface

Candidate code exposes `build_model(input_spec, output_spec)` and returns a `torch.nn.Module`. The Harness supplies specs derived from the active Research Problem Spec and the resolved candidate manifest.

For mask-only providers, tensor shorthand may be accepted when the output spec has one primary output. When auxiliary outputs are requested, models must return exactly the requested output keys and compatible tensors. The names and shapes are provider/spec-defined.

## Training policy

Candidate manifests may select only schema-supported and spec-allowed Harness-owned knobs. Implemented generic manifest fields include bounded learning rate, batch size, max epochs, scheduler policy, early stopping policy, sampling policy, frame-selection policy, augmentation policy, and optional provider-declared auxiliary targets.

Current scheduler policies are `constant_lr`, `cosine_decay`, and `reduce_on_plateau`. Early stopping is Harness-owned and records selection metric/mode, stop reason, completed epochs, and whether the best checkpoint was restored.

Future capability candidates include additional losses such as focal variants, additional optimizers such as SGD momentum, granular augmentation DSLs, weight decay controls, mixed precision controls, gradient clipping, and advanced sampling policies. They are not current generic manifest authority unless implemented in schema/code and exposed by the active Research Problem Spec.

## Run artifacts

Successful Harness-managed training runs produce the artifacts needed for comparison, diagnosis, and follow-up. The implemented trainer writes a best-validation checkpoint for successful runs at `outputs/models/best_epoch_model.pt` and records it in `outputs/best_metrics.json` and `outputs/final_metrics.json`.

Expected run artifacts include:

- `outputs/final_metrics.json` — final completed epoch/result summary.
- `outputs/best_metrics.json` — best validation epoch selected by the Research Problem selection metric, including `model_artifact`.
- `outputs/models/best_epoch_model.pt` — Harness-owned PyTorch checkpoint with epoch, selection metric/value, and CPU-readable `state_dict`.
- `outputs/metrics.jsonl` — per-step/per-epoch metric history.
- `outputs/model_summary.json` — smoke-test model summary.
- `resolved_manifest.yaml` — manifest after Harness defaults and validation.
- `run_metadata.json` — run status, timestamps, backend, provider provenance, and failure details where applicable.
- `outputs/prediction_samples/` — bounded qualitative artifacts when configured/supported.
- `outputs/logs/` — validation, smoke-test, training, timeout, and backend logs.

Rejected or failed runs should still produce clear metadata/logs when possible.

## Post-run evaluations

Post-Run Evaluations are run-scoped Harness operations. They do not create new Candidate Experiments or new Runs. Request-gated evaluations write under `outputs/evaluations/<evaluation_id>/` for the original Run and record ledger events. Current evaluation IDs derive from request IDs, for example `eval_<request_id>`.

## Future approved weights

Current code rejects candidate-supplied checkpoints and arbitrary runtime pretrained-weight downloads. Approved Weight Artifacts and Pretrained Weight Requests remain future architecture: they should be documented as follow-on design intent until a registry/workflow is implemented and tested.
