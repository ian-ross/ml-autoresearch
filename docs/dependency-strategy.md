# Dependency strategy

ML Autoresearch separates the host development environment, Research Workspace Root configuration, and runtime images used by Candidate Execution Boundary and Agent Control Boundary workflows.

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
- Docker runner images own the PyTorch/CUDA runtime used for Candidate Experiment smoke tests and training. The packaged runner recipe installs Python 3.12, installs the pinned PyTorch CUDA 12.1 wheel explicitly, installs only non-PyTorch runtime dependencies, and then installs `ml-autoresearch` with dependency resolution disabled, so package installation does not replace the pinned PyTorch/CUDA stack.

Candidate Experiments do not choose dependencies, Docker images, CUDA versions, or GPU access.

## Host development workflow

```bash
uv sync --python 3.12 --extra dev
uv run --python 3.12 pytest -q
```

Host tests are CPU-only by design. They should not initialize or depend on the host NVIDIA driver.

## Research Workspace dependency workflow

A Research Problem Repository is initialized separately from the reusable Harness package:

```bash
mkdir my-research-problem
cd my-research-problem
uv init --package --python 3.12
uv add ml-autoresearch
uv run ml-autoresearch setup \
  --problem-id my_research_problem \
  --provider-module my_research_problem.research_problem
```

The Research Workspace Root owns `ml-autoresearch.toml`. That Workspace Configuration records the active Research Problem provider, Candidate Execution Boundary policy, Agent Control Boundary policy, notification settings, and runtime image identities. Durable research state lives in the Research Problem Repository; generated runtime/build state lives under `.ml-autoresearch/`.

## Runtime image workflow

Build and validate runtime images from the Research Workspace Root or by passing `--workspace-root`:

```bash
uv run ml-autoresearch build-runtime-images --workspace-root . --update-config
uv run ml-autoresearch validate-runtime-images --workspace-root .
```

`build-runtime-images` stages packaged container build recipes under `.ml-autoresearch/container-build-recipes/`, prepares the Gondolin Agent Runtime Image assets under `.ml-autoresearch/images/agent/`, and builds a workspace- and Harness-version-specific Docker runner image tag for Candidate Execution Boundary runs. With `--update-config`, those two identities are written to `ml-autoresearch.toml`:

- `[agent_control_boundary].image` points to the Agent Runtime Image asset directory used by pi-fort/Gondolin.
- `[candidate_execution].docker_image` names the Docker runner image tag used for candidate smoke tests, training, and evaluation.

`validate-runtime-images` verifies that the configured Agent image assets and Docker runner metadata match the current Harness identity, workspace config, and optional development source override. It writes `.ml-autoresearch/runtime-images.validated.json`. Runtime command families such as `prepare-agent-boundary`, `autonomy-step`, `run-candidate`, and `evaluate-run` require a fresh validation stamp unless the operator uses an explicit skip option.

## Development source override

For local Harness development, a Research Workspace may set:

```toml
[runtime_images]
dev_source_path = "/path/to/ml-autoresearch"
```

This tells runtime-image build and validation to identify the editable Harness checkout rather than an installed package artifact. Rebuild and revalidate runtime images after changing:

- the Harness version or editable Harness checkout;
- `[runtime_images].dev_source_path`;
- packaged runtime image recipes or Harness dependencies;
- `ml-autoresearch.toml` fields that affect the Agent Runtime Image or Docker runner image identity.

A stale validation stamp is a safety signal that the configured runtime images may not match the Harness code that will orchestrate the run.

## Docker and cluster GPU validation

Before launching GPU-enabled training on the cluster, validate GPU visibility inside the same configured runner image:

```bash
uv run ml-autoresearch validate-docker-gpu --workspace-root .
```

The validation command runs the runner image with Docker GPU access and prints PyTorch version, container CUDA runtime version, driver-visible GPU information, and `torch.cuda.is_available()`. This checks the runtime environment that Docker-backed Candidate Execution Boundary runs will use; the host virtualenv is not the authoritative GPU probe.

If validation fails, check the host NVIDIA driver, Docker NVIDIA runtime configuration, and host driver compatibility with the container CUDA runtime. A newer container CUDA runtime requires a sufficiently new host NVIDIA driver.

Run CPU/no-GPU Docker training by default:

```bash
uv run ml-autoresearch run-candidate \
  --candidate candidates/my_candidate \
  --workspace-root . \
  --max-samples 8
```

Opt in to GPU access explicitly, or configure it in `ml-autoresearch.toml` for the workspace:

```bash
uv run ml-autoresearch run-candidate \
  --candidate candidates/my_candidate \
  --workspace-root . \
  --max-samples 8 \
  --docker-enable-gpu
```
