# ML Autoresearch Top-Level Plan

ML Autoresearch is a safe agent-assisted research system for proposing, running, evaluating, and iterating on ML model architecture experiments. The first Research Problem is Ground-Camera Contrail Detection using the GVCCS Dataset, but the infrastructure is intended to support other Research Problems later.

The project should be driven by the Research Loop: propose Candidate Experiments, run them safely, evaluate Results, and use those Results to guide the next Candidate Experiment.

## Shared vocabulary

- **ML Autoresearch** — the overall safe agent-assisted ML architecture exploration system.
- **Research Problem** — a target ML problem explored through Candidate Experiments.
- **Ground-Camera Contrail Detection** — the first Research Problem; binary semantic segmentation of contrail vs non-contrail pixels.
- **GVCCS Dataset** — the whole-sky-camera training dataset for Ground-Camera Contrail Detection.
- **Camera Domain Shift** — the known difference between GVCCS whole-sky-camera data and likely downstream conventional ground-camera imagery.
- **Candidate Experiment** — an agent-proposed runnable ML research package.
- **Candidate Experiment Contract** — the allowlisted interface for research variation without unsafe authority.
- **Harness** — the trusted implementation that owns training loops, data loading, validation, execution policy, and artifact persistence.
- **Candidate Experiment Runner** — the trusted subsystem that validates, executes, and records Candidate Experiments.
- **Run** — one execution attempt of a Candidate Experiment.
- **Result** — metrics and artifacts produced by a Run.

See `CONTEXT.md` for canonical definitions and relationships. Where this plan differs from the older `docs/project-brief.md`, this plan is canonical for implementation sequencing.

## Top-level approach

Use a Research Loop first approach:

1. Define Ground-Camera Contrail Detection as the first Research Problem.
2. Define the v1 Candidate Experiment Contract.
3. Build the simplest Harness capable of executing one useful baseline Candidate Experiment.
4. Add enough artifacts and metrics for the agent to learn from Results.
5. Add Docker, MLflow, async orchestration, and security hardening around the loop.
6. Expand the Candidate Experiment Contract only through Harness-owned parameters or audited capabilities.
7. Generalize to additional Research Problems after the contrail loop proves reusable seams.

## Top-level workstreams

### 1. Research Problem Definition

Formalize the first Research Problem:

- GVCCS dataset assumptions.
- binary semantic segmentation target: Contrail Mask for a Target Frame.
- v1 Input Modes: Single-Frame RGB Input and Centered Temporal RGB Clip Input.
- dataset splits.
- metrics.
- Camera Domain Shift note.

### 2. Candidate Experiment Contract

Specify the v1 allowlisted contract:

- `manifest.yaml` schema.
- `model.py` interface.
- input modes.
- output forms.
- primary and auxiliary losses.
- augmentation/data-policy allowlist.
- training knobs and resource bounds.
- Approved Weight Artifact references and Pretrained Weight Request workflow.

The contract minimizes unsafe authority, not research expressiveness.

### 3. Harness Core

Build the trusted core that owns:

- Candidate Experiment validation.
- Run directory lifecycle.
- resolved manifest generation.
- Run metadata.
- Run status model.
- artifact layout.
- rejection/block reasons.

### 4. Model Smoke Test

Implement validation before real training:

- import candidate model through the controlled interface.
- call `build_model(input_spec, output_spec)`.
- count parameters.
- run synthetic forward/backward.
- validate output names, shapes, and dtypes.

### 5. Training Implementation

Implement Harness-owned training:

- GVCCS data loader.
- fixed training loop.
- v1 loss allowlist: `bce_dice`, `focal_dice`, `focal_tversky`, plus auxiliary `focal_bce` / `weighted_bce`.
- v1 metrics: primary `val/dice`; secondary `val/iou`, `val/precision`, `val/recall`, `val/loss`.
- prediction sample generation.

### 6. Execution Boundary

Add the Candidate Execution Boundary. The current open issue sequence for this branch is #8-#13.

Policy decisions for the branch:

- Docker is the intended default execution backend once Docker synthetic training reaches parity with native execution.
- Native execution remains available as an explicit unsafe/developer backend.
- The first Docker issue proves structural containment only; stronger hardening is tracked separately.
- Harness-owned files stay at the Run root; operation-produced artifacts move under `outputs/`.
- In Docker, `candidate/`, `resolved_manifest.yaml`, and `run_metadata.json` are mounted read-only.
- In Docker, only `/outputs` and `/scratch` are writable.
- For GVCCS training, host `--data-root` is mounted read-only at `/data`.
- Candidate Experiments cannot request mounts or receive host dataset paths.
- Containers run with no network; stricter resource/time/security limits are added in the hardening issue.
- Wall-clock budget exhaustion should use a graceful shutdown protocol for training Runs: signal the training loop, allow a bounded grace period to write the best meaningful Result available, and force-terminate only if the grace period expires.
- The agent has no Docker or host shell access.

### 7. Observation Layer

Persist and expose Results:

- Harness uploads approved artifacts and metrics to MLflow.
- agent has read-only MLflow access.
- narrow commands for summaries and comparisons, e.g. `get_run_summary`, `get_best_runs`, `list_runs`.

### 8. Baseline Candidate Experiments

Provide initial candidates to exercise the loop:

- single-frame UNet baseline.
- temporal stack UNet baseline.
- optional auxiliary-head UNet using `line_logits` and/or `boundary_logits`.

## First tracer-bullet milestone

The first milestone is a tiny real-training vertical slice.

It should use native/local execution first, not Docker or MLflow, and should prove that the Candidate Experiment Contract and Harness can support the Research Loop.

### Success criteria

1. A sample Candidate Experiment with `manifest.yaml` and `model.py` is accepted.
2. The Harness validates the manifest against the v1 contract.
3. The Harness builds the model via `build_model(input_spec, output_spec)`.
4. The Harness runs a synthetic forward/backward smoke test.
5. The Harness trains on a deterministic synthetic contrail-like segmentation fixture for at least one epoch, then later swaps in a tiny GVCCS subset or fixture subset.
6. The Harness computes:
   - `val/dice`
   - `val/iou`
   - `val/precision`
   - `val/recall`
   - `val/loss`
7. The Harness writes required Run artifacts:
   - `final_metrics.json`
   - `metrics.jsonl`
   - `model_summary.json`
   - `resolved_manifest.yaml`
   - `run_metadata.json`
   - `prediction_samples/`
   - `logs/`

   The current native implementation writes operation artifacts at the Run root. Issue #8 will move operation artifacts under `outputs/` for the Docker boundary.
8. A human or agent can inspect the Run and decide what Candidate Experiment to try next.

### Explicit non-goals

The first tracer bullet does not need:

- Docker execution.
- MLflow persistence.
- asynchronous queueing.
- pretrained weight workflow.
- production security claims.
- full GVCCS training.

## Notes and constraints

- Implementation language is Python.
- ML implementation uses PyTorch.
- Candidate Experiment manifest validation uses Pydantic v2.
- The Harness always owns training loops and data loading.
- Candidate Experiments must not access arbitrary filesystem paths, network, Docker, dataset paths, or MLflow writes.
- Candidate Experiments may reference only Approved Weight Artifacts by stable ID.
- Runtime pretrained weight downloads are forbidden.
- Wall-Clock Budget Policy is Harness-owned and intentionally adjustable; early exploration may use small budgets to encourage many cheap architecture comparisons.
- GVCCS is whole-sky-camera data; downstream conventional-camera evaluation is separate from the initial ML Autoresearch loop.
- Real GVCCS data is not checked into this repository; tests use synthetic or GVCCS-like fixtures, and real training points at a local downloaded dataset path.
