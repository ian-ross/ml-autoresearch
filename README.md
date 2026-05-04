# ML Autoresearch

ML Autoresearch is a safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments.

The first Research Problem is **Ground-Camera Contrail Detection**: binary semantic segmentation of contrail pixels from ground-camera imagery using the GVCCS Dataset. The system is designed so future Research Problems can reuse the same Harness, Candidate Experiment Contract, Run lifecycle, and observation commands.

## Current status

This repository currently contains the local tracer-bullet Harness with a Docker-backed Candidate Execution Boundary:

- validate a local Candidate Experiment directory
- create Harness-owned Run directories
- smoke-test candidate models through the controlled `build_model(input_spec, output_spec)` interface
- train one epoch on deterministic synthetic data or a local GVCCS-like/GVCCS dataset
- for Docker GVCCS training, mount the host `--data-root` read-only at `/data` inside the container
- write local Run artifacts such as metrics, metadata, logs, model summaries, and prediction samples
- inspect local Runs without MLflow

The native backend remains available as an explicit developer-unsafe escape hatch. MLflow upload, async scheduling, and stronger production isolation are planned later layers around the same Research Loop.

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

This project uses Python 3.12. Host development and tests use a pinned CPU-only PyTorch build so local installs do not accidentally resolve a GPU/CUDA PyTorch stack. Host unit tests are CPU-only by design and should not initialize or depend on the host NVIDIA driver. With `uv` installed:

```bash
uv sync --python 3.12 --extra dev
uv run --python 3.12 pytest -q
```

The `uv.lock` file is resolved for the Python 3.12 project baseline; older Python interpreters are not supported. Base `ml-autoresearch` installs do not include PyTorch; use the `dev` extra for host tests. Docker-backed Candidate Execution Boundary runs get their runtime PyTorch/CUDA stack from the runner image, specifically the pinned `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` runner image, not from base package metadata. That image currently provides Python 3.11; the project Python 3.12 baseline applies to host development and package resolution until the Docker runtime issue is revisited with a Python 3.12-compatible CUDA 12.1 PyTorch image.

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

Build the pinned Docker runner image:

```bash
docker build -t ml-autoresearch-runner:local .
```

Run against a local GVCCS-like or real GVCCS data root using the default Docker backend:

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --data-root /path/to/GVCCS \
  --max-samples 8
```

Real GVCCS data is not committed to this repository. The Docker backend validates the host path and mounts it read-only at `/data`; Candidate Experiments cannot choose data paths or mounts. See [`docs/gvccs-data.md`](docs/gvccs-data.md) for expected local layout.

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
run_metadata.json          Run status, timestamps, sources, dataset metadata, artifact references
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
- Docker execution forms the current Candidate Execution Boundary for smoke tests and training
- future MLflow integration should be written by the trusted Harness, not candidate code

The current Harness is still a development tracer bullet, not a production sandbox.
