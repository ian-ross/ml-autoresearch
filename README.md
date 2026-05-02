# ML Autoresearch

ML Autoresearch is a safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments.

The first Research Problem is **Ground-Camera Contrail Detection**: binary semantic segmentation of contrail pixels from ground-camera imagery using the GVCCS Dataset. The system is designed so future Research Problems can reuse the same Harness, Candidate Experiment Contract, Run lifecycle, and observation commands.

## Current status

This repository currently contains the native/local tracer-bullet Harness:

- validate a local Candidate Experiment directory
- create Harness-owned Run directories
- smoke-test candidate models through the controlled `build_model(input_spec, output_spec)` interface
- train one epoch on deterministic synthetic data or a local GVCCS-like/GVCCS dataset
- write local Run artifacts such as metrics, metadata, logs, model summaries, and prediction samples
- inspect local Runs without MLflow

Docker execution, MLflow upload, async scheduling, and stronger production isolation are planned later layers around the same Research Loop. The current open Docker branch is tracked by issues #8-#13.

## Core concepts

- **Candidate Experiment**: an agent-proposed runnable ML research package.
- **Candidate Experiment Contract**: the allowlisted interface for research variation without unsafe authority.
- **Harness**: trusted code that owns validation, data loading, training, execution policy, and artifacts.
- **Run**: one execution attempt of a Candidate Experiment.
- **Result**: metrics and artifacts produced by a Run.

See [`CONTEXT.md`](CONTEXT.md) for canonical project vocabulary.

## Repository layout

```text
src/ml_autoresearch/        Python package and CLI
tests/                      unit/integration tests and small fixtures
docs/                       design notes, lifecycle docs, ADRs, and dataset notes
CONTEXT.md                  canonical domain language
```

Important docs:

- [`docs/top-level-plan.md`](docs/top-level-plan.md)
- [`docs/candidate-experiment-contract.md`](docs/candidate-experiment-contract.md)
- [`docs/run-lifecycle.md`](docs/run-lifecycle.md)
- [`docs/gvccs-data.md`](docs/gvccs-data.md)

## Development setup

This project uses Python and PyTorch. With `uv` installed:

```bash
uv sync --dev
uv run pytest -q
```

Run the CLI via:

```bash
uv run ml-autoresearch --help
# or
uv run python -m ml_autoresearch.cli --help
```

## Candidate Experiment contract

A minimal Candidate Experiment is a directory with:

```text
candidate/
├── manifest.yaml
└── model.py
```

Example manifest:

```yaml
name: single_frame_unet_baseline
description: Tiny single-frame mask-only baseline for harness validation.
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
```

`model.py` must expose:

```python
def build_model(input_spec: dict, output_spec: dict):
    ...
```

The Harness owns training loops, data loading, filesystem paths, artifact persistence, and run policy. Candidate Experiments must not provide arbitrary shell scripts, datasets, notebooks, checkpoints, Dockerfiles, custom data loaders, or MLflow logging code.

A working fixture lives at:

```text
tests/fixtures/candidates/single_frame_unet_baseline
```

## Running a Candidate Experiment

Run against deterministic synthetic fixture data:

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --synthetic-fixture
```

Run against a local GVCCS-like or real GVCCS data root:

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --data-root /path/to/GVCCS \
  --max-samples 8
```

Real GVCCS data is not committed to this repository. See [`docs/gvccs-data.md`](docs/gvccs-data.md) for expected local layout.

## Inspecting local Runs

Observation commands read only local `runs/` artifacts and do not require MLflow.

```bash
uv run ml-autoresearch list-runs --runs-root runs
uv run ml-autoresearch list-runs --runs-root runs --json

uv run ml-autoresearch run-summary --runs-root runs --run-id <run_id>
uv run ml-autoresearch run-summary --runs-root runs --run-id <run_id> --json

uv run ml-autoresearch get-best-runs --runs-root runs
uv run ml-autoresearch get-best-runs --runs-root runs --metric val/dice --limit 10 --json
```

`get-best-runs` ranks completed Runs by `val/dice` by default.

## Local Run artifacts

A completed local Run reserves `runs/<run_id>/` for Harness-owned files plus an `outputs/` directory:

```text
candidate/                 copied Candidate Experiment source
resolved_manifest.yaml     Harness-normalized manifest
run_metadata.json          Run status, timestamps, sources, artifact references
outputs/
  model_summary.json       model parameter summary
  metrics.jsonl            batch/epoch metric stream
  final_metrics.json       final Result metrics, including val/dice
  prediction_samples/      qualitative PNG samples and samples.json
  logs/                    validation, smoke-test, and training logs
```

Rejected, smoke-failed, and failed Runs are also represented as Run directories with metadata and `outputs/logs/` so humans and agents can inspect repair feedback.

## Safety model

The intended architecture separates authority:

- agents propose Candidate Experiments through a narrow contract
- the trusted Harness validates, runs, and records them
- future Docker execution will form the Candidate Execution Boundary
- future MLflow integration should be written by the trusted Harness, not candidate code

The current native/local Harness is a development tracer bullet, not a production sandbox.
