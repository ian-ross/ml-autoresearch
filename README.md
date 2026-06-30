# ML Autoresearch

Research campaign events are recorded in the canonical append-only `research-ledger.jsonl` file through the Harness-owned `record-research-event` CLI/API. See `docs/campaign-autonomy-architecture.md`.

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
- inspect local Runs through Harness-owned metadata and artifacts

The native backend remains available as an explicit developer-unsafe escape hatch. Async scheduling and stronger production isolation are planned later layers around the same Research Loop.

## Core concepts

- **Candidate Experiment**: an agent-proposed runnable ML research package.
- **Candidate Experiment Contract**: the allowlisted interface for research variation without unsafe authority.
- **Harness**: trusted code that owns validation, data loading, training, execution policy, and artifacts.
- **Run**: one execution attempt of a Candidate Experiment.
- **Result**: metrics and artifacts produced by a Run.

See [`CONTEXT.md`](CONTEXT.md) for canonical project vocabulary.

## Repository layout

This repository is the reusable **Harness** package. It should not contain live Research Loop state for a specific problem. A separate **Research Problem Repository** is the **Research Workspace Root** for one problem; it contains `ml-autoresearch.toml`, durable research memory, Candidate Experiments, Research Notes, and generated agent/workspace state.

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
- [`docs/gvccs-features.md`](docs/gvccs-features.md) — temporary GVCCS notes pending migration to the GVCCS Research Problem repository
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

The `uv.lock` file is resolved for the Python 3.12 project baseline; older Python interpreters are not supported. Base `ml-autoresearch` installs do not include PyTorch; use the `dev` extra for host tests. Docker-backed Candidate Execution Boundary runs get their runtime PyTorch/CUDA stack from the workspace-specific runner image and also use Python 3.12, with PyTorch `2.5.1+cu121` installed from the PyTorch CUDA 12.1 wheel index on top of the pinned `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` base image. Use `ml-autoresearch build-runtime-images` and `ml-autoresearch validate-runtime-images` from a Research Workspace Root to stage/build the packaged runtime recipes and write the workspace validation stamp.

Run the CLI via:

```bash
uv run ml-autoresearch --help
# or
uv run python -m ml_autoresearch.cli --help
```

## Creating a Research Workspace Root

A new Research Problem starts in its own trusted repository. Initialize the Python project, add the reusable Harness package, then run setup to create the Workspace Configuration and initial research-memory layout:

```bash
mkdir my-research-problem
cd my-research-problem
uv init --package --python 3.12
uv add ml-autoresearch
uv run ml-autoresearch setup \
  --problem-id my_research_problem \
  --provider-module my_research_problem.research_problem
```

`setup` writes `ml-autoresearch.toml`, starter Research Problem materials, `EXPERIMENT_INDEX.md`, `research-ledger.jsonl`, canonical handoff directories, and Agent Workspace directories. Durable research state belongs in the Research Problem Repository. Generated operational state belongs under `.ml-autoresearch/`, `agent-work/`, `agent-history/`, `agent-reference/`, and `agent-research-problem/`.

Build and validate workspace-specific runtime images before boundary, execution, or autonomy workflows:

```bash
uv run ml-autoresearch build-runtime-images --workspace-root . --update-config
uv run ml-autoresearch validate-runtime-images --workspace-root .
```

The build command records two different runtime identities in `ml-autoresearch.toml`: the Gondolin Agent Runtime Image asset directory, usually `.ml-autoresearch/images/agent/`, and the Docker runner image tag used by Candidate Execution Boundary runs. The validation command writes `.ml-autoresearch/runtime-images.validated.json`; runtime commands reject stale stamps unless the operator explicitly opts out.

For Harness source development, `[runtime_images].dev_source_path` can identify an editable Harness checkout. Rebuild and revalidate runtime images after changing the Harness dependency, the development source override, packaged container recipes, or any workspace config fields that affect image identity.

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

The Harness owns training loops, data loading, filesystem paths, artifact persistence, and run policy. Candidate Experiments may select only allowlisted Harness/provider-owned data policy values such as `data.sampling_policy`; they must not provide arbitrary shell scripts, datasets, notebooks, checkpoints, Dockerfiles, custom data loaders, samplers, transforms, external persistence, or runtime weight-fetching code.

A working fixture lives at:

```text
tests/fixtures/candidates/single_frame_unet_baseline
```

## Running a Candidate Experiment

Run against the Research Problem provider configured in the Research Workspace Root's `ml-autoresearch.toml`. `run-candidate` defaults to autonomous-mode proposal validation; the committed Harness test fixture has no `PROPOSAL.md`, so this fixture example uses `--no-require-proposal`.

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --workspace-root /path/to/research-workspace \
  --no-require-proposal
```

For a real Research Problem Repository, prefer running from the workspace root and relying on `ml-autoresearch.toml` for the provider, backend, Docker image, GPU policy, runs root, ledger path, and prediction-sample defaults:

```bash
cd /path/to/research-workspace
uv run ml-autoresearch run-candidate \
  --candidate candidates/my_candidate \
  --workspace-root .
```

Run with explicit policy overrides while reusing the same configured Research Problem provider:

```bash
uv run ml-autoresearch run-candidate \
  --candidate candidates/my_candidate \
  --workspace-root . \
  --max-samples 8 \
  --prediction-sample-policy adjacent_and_scattered
```

Research Problem-specific trusted code is loaded through the filesystem provider configuration in `ml-autoresearch.toml`, not by Candidate Experiments. The configuration names the Research Problem id, local package root, provider target such as `gvccs.research_problem:build_spec`, expected contract version, and data configuration such as the dataset root. The Harness validates the returned Spec, records provider provenance in Run metadata, and then uses only the checked Research Problem adapter interface. Agent Control Boundary setup generates a curated read-only `agent-research-problem/` snapshot mounted at `/research-problem` with importable provider Python sources, declared Research Problem Brief documents, declared Dataset Profile Artifacts, and the generated index, while excluding undeclared package resources and raw datasets. It also exposes the progressive-disclosure index in `agent-work/AGENTS.md` and `agent-work/RESEARCH_PROBLEM_BRIEF_INDEX.md`, with document roles, summaries, and `/research-problem/...` read commands rather than embedding full brief content in every prompt or mounting the full Research Problem Repository into the agent environment. Agent handoff/autonomy flows require this explicit provider configuration and fail clearly instead of using a built-in/default Research Problem fallback.

To run GVCCS-linked example/integration tests, set `ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT` and `ML_AUTORESEARCH_TEST_PROBLEM_ROOT` to your external package roots before invoking the test suite.

Prediction Sample Policy is a Run-level Harness option, not a Candidate Experiment manifest option. Supported values are `first_n` (default) and Research Problem-provided policies such as `adjacent_and_scattered` in the GVCCS example package.

Docker GPU access is disabled by default. Before launching GPU-enabled Candidate Experiment training on the cluster, validate GPU visibility inside the same runner image used for candidate execution:

```bash
uv run ml-autoresearch validate-docker-gpu
```

The command runs the pinned runner image with `--gpus all` and prints `torch.__version__`, `torch.version.cuda`, the driver-visible GPU name, and `torch.cuda.is_available()` from inside the container. It does not mount Candidate Experiment code, data, or run outputs, and is safe to run on a cluster GPU node before training. If validation fails, check that the host NVIDIA driver is new enough for the container CUDA runtime and that Docker's NVIDIA runtime is available; do not use the host virtualenv as the authoritative GPU probe for Docker-backed Runs.

Opt in explicitly for Docker runs when running on a GPU-capable host or cluster node:

```bash
uv run ml-autoresearch run-candidate \
  --candidate candidates/my_candidate \
  --workspace-root . \
  --max-samples 8 \
  --docker-enable-gpu
```

## Inspecting local Runs

Observation commands read only local `runs/` artifacts and do not require external tracking services.

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
- any future external tracking integration should be written by the trusted Harness, not candidate code

The current Harness is still a development tracer bullet, not a production sandbox.
