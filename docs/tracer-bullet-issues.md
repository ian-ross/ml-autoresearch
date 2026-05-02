# Tracer-Bullet Issue Breakdown

This document refines `docs/top-level-plan.md` into independently grabbable implementation issues. The first milestone is a local/native Research Loop: submit a Candidate Experiment, validate it, create a Run, smoke-test the model, train on a deterministic synthetic contrail-like fixture, train on a local GVCCS-compatible data path, generate artifacts, and inspect Results.

Where this document differs from `docs/project-brief.md`, the top-level plan and this breakdown are canonical for implementation sequencing.

## Milestone: local Research Loop

Non-goals for this milestone:

- Docker execution.
- MLflow persistence.
- async queueing.
- pretrained weight workflow.
- production security claims.
- full GVCCS training.
- temporal clips or auxiliary heads.

### Issue 1: Candidate Experiment filesystem contract and sample fixture

Create the minimal Python project skeleton and define the initial Candidate Experiment contract for a local directory submission.

Scope:

- Add `pyproject.toml`.
- Add package directory, e.g. `ml_autoresearch/`.
- Add `tests/`.
- Declare initial dependencies: Pydantic v2, PyYAML, pytest.
- Define candidate layout:

  ```text
  candidate/
  ├── manifest.yaml
  └── model.py
  ```

- Add a sample Candidate Experiment fixture using Single-Frame RGB Input and mask-only output.
- Include a real PyTorch-looking `model.py` exposing:

  ```python
  def build_model(input_spec, output_spec):
      ...
  ```

  but do not import or execute it yet.
- Implement Pydantic v2 manifest model and YAML loading.
- Validate the minimal manifest:

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

Acceptance criteria:

- Required files `manifest.yaml` and `model.py` are detected.
- Candidate source is a directory path only; zip/tar/archive submission is unsupported.
- Missing required manifest fields are rejected with clear reasons.
- Unknown `input_mode`, `output_form`, loss, or optimizer are rejected.
- Scalar training values are type/range checked.
- Validator returns a normalized manifest object or a clear rejection.
- Required fields and allowed values are covered by tests.
- Extra `.py` helper files and `README.md` may be allowed, but the sample fixture stays minimal.
- Symlinks, hidden files, checkpoints (`*.pt`, `*.pth`, `*.ckpt`), archives, shell scripts, notebooks, dataset files, and arbitrary config blobs are rejected.

Out of scope:

- Model import/execution.
- PyTorch dependency execution in tests.
- Run directory creation.
- Training.
- Docker or MLflow.

### Issue 2: Run directory lifecycle and candidate copy/quarantine

Create a Run for accepted candidates and persist the source candidate plus Harness-owned metadata.

Scope:

- Generate Harness-owned Run IDs with sortable timestamp plus short random suffix, e.g. `run_20260502_153000_ab12cd`.
- Create run directory layout:

  ```text
  runs/
  └── run_YYYYMMDD_HHMMSS_<suffix>/
      ├── candidate/
      │   ├── manifest.yaml
      │   └── model.py
      ├── resolved_manifest.yaml
      ├── run_metadata.json
      └── logs/
          └── validation.log
  ```

- Copy accepted candidate source into `candidate/` under the Run directory.
- Write `resolved_manifest.yaml` after validation/default normalization.
- Write initial `run_metadata.json`.
- Record accepted/rejected status and clear rejection reasons.
- Introduce minimal CLI:

  ```bash
  ml-autoresearch submit-candidate --candidate path/to/candidate --runs-root runs
  ```

Status model to reserve:

- `accepted`
- `rejected`
- `smoke_testing`
- `smoke_failed`
- `training`
- `completed`
- `failed`

Acceptance criteria:

- Accepted candidate creates exactly one unique Run directory.
- Candidate source is copied into the Run directory, not executed in-place.
- `resolved_manifest.yaml` records the Harness-resolved manifest, not the candidate source manifest verbatim unless no defaults changed.
- `run_metadata.json` includes Run ID, status, timestamps, candidate source information, Harness/package version if available, and rejection reason when relevant.
- Rejected candidates produce clear CLI output and logs; implementation may either avoid creating a Run or create a rejected Run, but behavior must be documented and tested.
- CLI exits non-zero on rejected candidates.

Out of scope:

- Model import/execution.
- Training.
- Docker or MLflow.

### Issue 3: PyTorch model import and synthetic smoke test

Import the copied candidate model from the Run directory and perform a cheap synthetic PyTorch validation before real training.

Scope:

- Add PyTorch dependency.
- Import `candidate/model.py` from the Run directory through a controlled import path.
- Require `build_model(input_spec, output_spec)`.
- Use tracer-bullet specs:

  ```python
  input_spec = {
      "mode": "single_frame_rgb",
      "shape": [3, 128, 128],
  }

  output_spec = {
      "form": "mask_logits",
      "shape": [1, 128, 128],
  }
  ```

- Instantiate the model.
- Count parameters.
- Run synthetic forward pass with:

  ```python
  x = torch.randn(2, 3, 128, 128)
  target = torch.rand(2, 1, 128, 128)
  ```

- Support both v1 mask-only output forms:
  - raw tensor `Tensor[B, 1, H, W]`
  - dict `{"mask_logits": Tensor[B, 1, H, W]}`
- Reject auxiliary keys until the auxiliary-head issue.
- Validate output shape, dtype, and names.
- Run one backward pass with dummy loss.
- Write `model_summary.json`.
- Write `logs/smoke_test.log`.
- Update `run_metadata.json` to `smoke_testing`, then `accepted` or `smoke_failed`.

Acceptance criteria:

- Valid sample candidate passes smoke test.
- Import errors, missing `build_model`, bad output names, bad shapes, bad dtypes, backward failures, and parameter-budget violations are recorded as `smoke_failed` with clear reasons.
- Smoke failures create/keep a Run ID and artifacts for agent repair feedback.
- `model_summary.json` includes parameter count and input/output contract details.

Out of scope:

- GVCCS loading.
- Training loop.
- Docker or MLflow.

### Issue 4: Synthetic contrail-like fixture training loop

Implement the first Harness-owned training path using deterministic generated data before introducing GVCCS parsing.

Scope:

- Add fixed Harness-owned train/validation loop.
- Generate deterministic synthetic contrail-like segmentation data:
  - dark/blue-ish sky-like RGB background;
  - 1–3 thin bright diagonal/curved line segments as contrails;
  - binary masks matching those line segments;
  - optional noise/cloud-ish blobs as negatives.
- Use deterministic seed for reproducibility.
- Implement BCE+Dice loss for `bce_dice`.
- Train for one epoch on small generated train/val sets.
- Compute v1 scalar metrics:
  - `val/dice`
  - `val/iou`
  - `val/precision`
  - `val/recall`
  - `val/loss`
- Write `metrics.jsonl` and `final_metrics.json`.
- Write `logs/training.log`.
- Update status to `training`, then `completed` or `failed`.
- Extend CLI, for example:

  ```bash
  ml-autoresearch run-candidate --candidate path/to/candidate --runs-root runs --synthetic-fixture
  ```

Acceptance criteria:

- Valid sample candidate trains for at least one epoch on the synthetic fixture.
- Metrics files are written with the required metric keys.
- Failed training records clear status/reason in metadata and logs.
- CLI runs the full local path through validation, Run creation, smoke test, and synthetic training synchronously.
- Tests cover metric computations on simple known masks.

Out of scope:

- Real GVCCS loading.
- Prediction sample PNGs unless trivial and cheap.
- Docker or MLflow.

### Issue 5: Tiny GVCCS data adapter and GVCCS-like fixture

Add the first Harness-owned data adapter for Single-Frame RGB Input without checking real GVCCS data into the repository.

Scope:

- Define expected local GVCCS layout/config based on the downloaded dataset.
- Add `tests/fixtures/gvccs_like/` with 2–4 generated image/mask pairs that mimic the expected layout.
- Implement loader for RGB image and binary Contrail Mask pairs.
- Support local real data via CLI option such as:

  ```bash
  --data-root /path/to/gvccs
  ```

- Fail clearly if `--data-root` is missing or malformed.
- Discover image/mask pairs.
- Use deterministic train/val split.
- Resize/crop to `[128, 128]`.
- Run one epoch on bounded sample count, e.g. `--max-samples N`.
- Preserve synthetic fixture path and tests.

Acceptance criteria:

- No real GVCCS samples are checked into the repository.
- GVCCS-like fixture tests pass.
- Loader can train the sample candidate for one epoch against the fixture.
- Real local `--data-root` errors are understandable when layout/data is absent or malformed.
- Single-Frame RGB Input only.

Out of scope:

- Temporal clips.
- metadata-rich sampling.
- camera-domain tags.
- cloud-heavy negatives.
- full dataset preprocessing.
- Docker or MLflow.

### Issue 6: Prediction sample artifact generation

Generate qualitative artifacts so humans and agents can inspect model behavior, not just scalar metrics.

Scope:

- After validation/training, write `prediction_samples/`.
- Store both PNG files and a JSON manifest:

  ```text
  prediction_samples/
  ├── samples.json
  ├── sample_000_input.png
  ├── sample_000_ground_truth.png
  ├── sample_000_prediction.png
  ├── sample_000_overlay.png
  └── ...
  ```

- Work for synthetic fixture and GVCCS-like/real loader.
- Keep sample count bounded/configurable.
- Record sample paths in `samples.json` and reference them from `run_metadata.json` or `final_metrics.json`.
- Include cheap per-sample metadata where available: sample ID, split, dice/iou, source image path if allowed.

Acceptance criteria:

- Completed Runs include bounded prediction sample artifacts.
- PNGs are readable and aligned in size.
- `samples.json` references existing files.
- Artifact generation failures are logged clearly and either fail the Run or are represented by an explicit artifact status.

Out of scope:

- MLflow upload.
- advanced failure mining or hard-negative selection.

### Issue 7: Local observation commands

Complete the local Research Loop by making Runs inspectable without MLflow.

Scope:

- Add Python API functions for reading local Run artifacts.
- Add CLI wrappers for:
  - `list-runs`
  - `get-run-summary` or `run-summary`
  - `get-best-runs`
- Read only local `runs/` artifacts.
- Sort best runs by `val/dice` by default.
- Support `--json` output.
- Skip or report missing/corrupt runs clearly.

Acceptance criteria:

- A human or agent can run a candidate, inspect its summary, list all local Runs, and identify best Runs by `val/dice`.
- CLI presentation is thin over tested Python functions.
- Commands handle rejected, smoke-failed, failed, and completed Runs.
- Observation commands do not require MLflow.

Out of scope:

- MLflow persistence/querying.
- async orchestration.

## Next branch: Candidate Execution Boundary with Docker

After the local Research Loop issues are complete, the next priority is the Docker-backed Candidate Execution Boundary. The approved open issue sequence is:

### #8: Separate Harness-owned Run files from operation outputs

Prepare the Run layout for Docker by moving operation-produced artifacts under `outputs/` while keeping Harness-owned files at the Run root:

```text
runs/run_x/
├── candidate/                 # Harness-owned source copy
├── resolved_manifest.yaml      # Harness-owned
├── run_metadata.json           # Harness-owned
└── outputs/                    # operation-produced artifacts
    ├── logs/
    ├── model_summary.json
    ├── metrics.jsonl
    ├── final_metrics.json
    └── prediction_samples/
```

### #9: Add execution backend abstraction and Docker smoke-test backend

- Introduce `ExecutionBackend` with native and Docker implementations.
- Add `--backend native|docker` and `--docker-image` with default `ml-autoresearch-runner:local`.
- Commit a Dockerfile and an in-container smoke-test entrypoint.
- Use structural containment only:
  - `/candidate:ro`
  - `/resolved_manifest.yaml:ro`
  - `/run_metadata.json:ro`
  - `/outputs:rw`
  - `/scratch:rw`
  - `--network none`
- Host Harness remains sole writer of Run metadata.
- Native backend is retained as developer-unsafe.

### #10: Add Docker synthetic training backend and flip default backend to Docker

- Extend `ExecutionBackend` for synthetic training.
- Run validation, smoke test, and synthetic training through Docker.
- Synthetic training uses no data mount.
- After Docker synthetic parity, `run-candidate` defaults to Docker and `--backend native` remains the explicit unsafe/developer escape hatch.

### #11: Add Docker GVCCS training backend with read-only data mount

- Extend Docker execution for local GVCCS-compatible data training.
- Host `--data-root` is mounted read-only at `/data`.
- Harness-owned metadata records dataset id `gvccs`, real host path, and container path `/data`.
- Candidate Experiments cannot request mounts or receive host dataset paths.

### #12: Harden the Docker Candidate Execution Boundary

- Run as non-root.
- Use read-only root filesystem where practical.
- Keep `/outputs` and `/scratch` as the only writable paths.
- Implement wall-clock budget handling as graceful shutdown, not only a hard container kill.
- On budget exhaustion, the Harness signals the in-container training loop and gives it a bounded grace period to finish a safe unit of work, write logs/metrics, and persist the best meaningful Result available.
- If the grace period expires, the Harness force-terminates the container and records that forced timeout failure clearly.
- Run metadata distinguishes normal completion, graceful timeout completion, and forced timeout failure while remaining Harness-owned.
- Add CPU/memory/GPU policy.
- Use environment allowlist.
- Drop capabilities and add process limits where practical.
- Assert no network, no privileged mode, no Docker socket, no host project mount, and no arbitrary filesystem mounts.

This issue is mandatory before making strong production safety claims.

### #13: Add container image build workflow

- Keep the manual local build path initially:

  ```bash
  docker build -t ml-autoresearch-runner:local .
  ```

- Add a repeatable local build helper for the local runner image.
