# Dependency strategy

ML Autoresearch separates the host development environment from the Docker runtime used by Candidate Execution Boundary runs.

## Baselines

- Host development and package resolution use **Python 3.12** (`requires-python = ">=3.12,<3.13"`).
- Host unit tests use pinned **CPU-only PyTorch** via the `dev` extra.
- Docker-backed Candidate Experiment execution targets a pinned **CUDA 12.1** runner stack: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`, Python 3.12, and PyTorch `2.5.1+cu121` from the PyTorch CUDA 12.1 wheel index.
- The Docker runner target is **CUDA 12.1** and uses the same Python 3.12 baseline as host development.

## Why base installs do not include PyTorch

The base dependencies do not include PyTorch. This is intentional: a broad dependency such as `torch>=2,<3` can cause local installs to resolve a large or incompatible GPU/NVIDIA stack based on the host environment. That is not the authority boundary we want.

Instead:

- base `ml-autoresearch` installs contain the Harness code and non-runtime dependencies;
- host development installs use `uv sync --python 3.12 --extra dev`, which resolves the pinned CPU-only PyTorch wheel from the PyTorch CPU index;
- Docker runner images own the PyTorch/CUDA runtime used for Candidate Experiment smoke tests and training. The Dockerfile installs Python 3.12, installs the pinned PyTorch CUDA 12.1 wheel explicitly, installs only non-PyTorch runtime dependencies, and then installs `ml-autoresearch` with dependency resolution disabled, so package installation does not replace the pinned PyTorch/CUDA stack.

Candidate Experiments do not choose dependencies, Docker images, CUDA versions, or GPU access.

## Host development workflow

```bash
uv sync --python 3.12 --extra dev
uv run --python 3.12 pytest -q
```

Host tests are CPU-only by design. They should not initialize or depend on the host NVIDIA driver.

## Docker and cluster workflow

Build the runner image once on a Docker-capable host or cluster node. Docker-backed Runs use `ml-autoresearch-runner:local` by default, and the preferred repeatable local workflow is:

```bash
make runner-image
```

The manual Docker command remains a valid fallback and builds the same default image:

```bash
docker build -t ml-autoresearch-runner:local .
```

Before launching GPU-enabled training on the cluster, validate GPU visibility inside the same runner image:

```bash
uv run ml-autoresearch validate-docker-gpu
```

The validation command runs the runner image with Docker GPU access and prints PyTorch version, container CUDA runtime version, driver-visible GPU information, and `torch.cuda.is_available()`. This checks the runtime environment that Docker-backed Candidate Execution Boundary runs will use; the host virtualenv is not the authoritative GPU probe.

If validation fails, check the host NVIDIA driver, Docker NVIDIA runtime configuration, and host driver compatibility with the container CUDA runtime. A newer container CUDA runtime requires a sufficiently new host NVIDIA driver.

Run CPU/no-GPU Docker training by default:

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --data-root /path/to/GVCCS \
  --max-samples 8
```

Opt in to GPU access explicitly:

```bash
uv run ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root runs \
  --data-root /path/to/GVCCS \
  --max-samples 8 \
  --docker-enable-gpu
```

Run metadata records whether Docker GPU policy was `disabled_by_default` or `enabled_by_harness_configuration`.
