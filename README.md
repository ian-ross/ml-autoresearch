# ML Autoresearch

Research campaign events are recorded in the canonical append-only `research-ledger.jsonl` file through the Harness-owned `record-research-event` CLI/API. See `docs/design/autonomous-research-campaign-plan.md`.

ML Autoresearch is a safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments.

The first Research Problem is **Ground-Camera Contrail Detection**: binary semantic segmentation of contrail pixels from ground-camera imagery using the GVCCS Dataset. The system is designed so future Research Problems can reuse the same Harness, Candidate Experiment Contract, Run lifecycle, and observation commands.

## Current status

This repository currently contains the local tracer-bullet Harness with a Docker-backed Candidate Execution Boundary:

- validate a local Candidate Experiment directory
- create Harness-owned Run directories
- smoke-test candidate models through the controlled `build_model(input_spec, output_spec)` interface
- train one epoch on deterministic synthetic data or the configured trusted Research Problem data
- for Docker Research Problem training, mount the trusted Research Problem package read-only and mount the host data root read-only at `/data` when configured
- write local Run artifacts such as metrics, metadata, logs, model summaries, and prediction samples
- record validated Research Ledger events for proposals, candidates, Runs, evaluations, capability requests, reports, and pauses
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
- [`docs/dependency-strategy.md`](docs/dependency-strategy.md)

## Development setup

This project uses Python 3.12. Host development and tests use a pinned CPU-only PyTorch build so local installs do not accidentally resolve a GPU/CUDA PyTorch stack. Host unit tests are CPU-only by design and should not initialize or depend on the host NVIDIA driver. In order for `uv` to work inside the sandbox VM used with Pi (via `pi-fort`), the virtual environment needs to be self-contained. With `uv` installed:

```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
uv python install 3.12
uv venv --managed-python --python 3.12 --relocatable
uv sync --managed-python --extra dev
uv run pytest -q
```

The `uv.lock` file is resolved for the Python 3.12 project baseline; older Python interpreters are not supported. Base `ml-autoresearch` installs do not include PyTorch; use the `dev` extra for host tests. Docker-backed Candidate Execution Boundary runs get their runtime PyTorch/CUDA stack from the runner image and also use Python 3.12, with PyTorch `2.5.1+cu121` installed from the PyTorch CUDA 12.1 wheel index on top of the pinned `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` base image. `containers/Dockerfile.runner` installs `ml-autoresearch` without dependency resolution so the pinned PyTorch/CUDA stack is not replaced. Build it with `make -C containers runner-image` or the equivalent manual fallback `docker build -f containers/Dockerfile.runner -t ml-autoresearch-runner:local .`.

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

A Candidate source may also include:

- `README.md`: narrative notes for the candidate owner.
- `PROPOSAL.md`: autonomous-mode experiment rationale required by the default `--require-proposal` validation on `submit-candidate` and `run-candidate`.

`Research Note` documents are distinct: they are captured outside candidate source directories and summarize Run outcomes in campaign context.

Example manifest:

```yaml
name: single_frame_unet_baseline
description: Tiny single-frame mask-only baseline for harness validation.
input_mode: single_frame_rgb
output_form: mask_logits
data:
  sampling_policy: sequential
  augmentation_policy: none
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

The Harness owns training loops, data loading, filesystem paths, artifact persistence, and run policy. Candidate Experiments may select only allowlisted Harness-owned data policy values such as `data.sampling_policy`; they must not provide arbitrary shell scripts, datasets, notebooks, checkpoints, Dockerfiles, custom data loaders, samplers, transforms, or MLflow logging code.

A working fixture lives at:

```text
tests/fixtures/candidates/single_frame_unet_baseline
```

## Running a Candidate Experiment

Run against deterministic synthetic fixture data. `run-candidate` defaults to autonomous-mode proposal validation; the committed test fixture has no `PROPOSAL.md`, so these fixture examples use `--no-require-proposal`.

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --synthetic-fixture \
  --no-require-proposal
```

Build the pinned Docker runner image. The Docker backend defaults to the local runner image tag `ml-autoresearch-runner:local`; the preferred repeatable local workflow is:

```bash
make runner-image
```

The manual Docker command remains a valid fallback and builds the same default image:

```bash
docker build -t ml-autoresearch-runner:local .
```

Run against the configured Research Problem data root using the default Docker backend. The current committed example is the initial Ground-Camera Contrail Detection Research Problem with a local GVCCS-like or real GVCCS data root:

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --data-root /path/to/GVCCS \
  --max-samples 8 \
  --prediction-sample-policy adjacent_and_scattered \
  --no-require-proposal
```

Research Problem-specific trusted code is loaded through a filesystem provider configuration, not by Candidate Experiments. A configuration names the Research Problem id, the local package root, the provider target such as `gvccs.research_problem:build_spec`, the expected contract version, and data configuration such as the dataset root. The Harness validates the returned Spec, records provider provenance in Run metadata, and then uses only the checked Research Problem adapter interface. The GVCCS example package lives outside this Harness repository at `/home/iross/code/gvccs-research-problem`; any GVCCS-named Python wrappers are legacy compatibility shims around generic Research Problem dispatch.

Prediction Sample Policy is a Run-level Harness option, not a Candidate Experiment manifest option. Supported values are `first_n` (default) and `adjacent_and_scattered` for the GVCCS example adapter.

Docker GPU access is disabled by default. Before launching GPU-enabled Candidate Experiment training on the cluster, validate GPU visibility inside the same runner image used for candidate execution:

```bash
uv run ml-autoresearch validate-docker-gpu
```

The command runs the pinned runner image with `--gpus all` and prints `torch.__version__`, `torch.version.cuda`, the driver-visible GPU name, and `torch.cuda.is_available()` from inside the container. It does not mount Candidate Experiment code, data, or run outputs, and is safe to run on a cluster GPU node before training. If validation fails, check that the host NVIDIA driver is new enough for the container CUDA runtime and that Docker's NVIDIA runtime is available; do not use the host virtualenv as the authoritative GPU probe for Docker-backed Runs.

Opt in explicitly for Docker runs when running on a GPU-capable host or cluster node:

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --data-root /path/to/GVCCS \
  --max-samples 8 \
  --docker-enable-gpu \
  --no-require-proposal
```

Real GVCCS data is not committed to this repository. The Docker backend validates the configured Research Problem data path, mounts it read-only at `/data`, and mounts the trusted provider package read-only; Candidate Experiments cannot choose data paths, provider packages, or mounts. See [`docs/gvccs-data.md`](docs/gvccs-data.md) for the GVCCS example layout.

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
  final_metrics.json       final completed epoch Result metrics
  best_metrics.json        best validation epoch metrics selected by max val/dice
  models/best_epoch_model.pt
                            Harness-owned weights for the best validation epoch
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
