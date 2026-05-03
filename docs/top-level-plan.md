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
5. Run Human-Guided Research Iterations for Ground-Camera Contrail Detection with the current limited contract.
6. Capture each iteration in a lightweight Markdown Research Note under `research-notes/`, and use Research Notes plus constraints as the input to later Experiment Proposals.
7. Use those iterations to identify what the autonomous Pi-agent proposal loop actually needs.
8. Expand the Candidate Experiment Contract only when Results show that a missing Harness-owned capability is blocking useful research.
9. Keep Research Problem-specific code and artifacts coupled to the local Harness until Human-Guided Research Iterations reveal concrete seams worth extracting.
10. Add Docker, MLflow, async orchestration, and security hardening around the loop as operational needs arise.
11. Generalize to additional Research Problems after the contrail loop proves reusable seams.

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

Add the Candidate Execution Boundary. Docker smoke testing, synthetic training, GVCCS training, and the initial hardening pass are now implemented for the local tracer-bullet Harness; remaining work in this area focuses on follow-on capabilities such as image build workflow, async scheduling, and production isolation beyond Docker hardening.

Policy decisions for the branch:

- Docker is the default execution backend for `run-candidate`.
- Native execution remains available as an explicit unsafe/developer backend.
- Harness-owned files stay at the Run root; operation-produced artifacts move under `outputs/`.
- In Docker, `candidate/`, `resolved_manifest.yaml`, and `run_metadata.json` are mounted read-only.
- In Docker, only `/outputs` and `/scratch` are writable container paths; `/outputs` is a run-scoped artifact mount and `/scratch` is bounded tmpfs.
- For GVCCS training, host `--data-root` is mounted read-only at `/data`.
- Candidate Experiments cannot request mounts or receive host dataset paths.
- Containers run with no network, non-root users, read-only root filesystems, dropped Linux capabilities, process/CPU/memory limits, and GPU disabled by default unless enabled by Harness configuration.
- Wall-clock budget exhaustion should use a graceful shutdown protocol for training Runs: signal the training loop, allow a bounded grace period to write the best meaningful Result available, and force-terminate only if the grace period expires.
- The agent has no Docker or host shell access.

### 7. Observation Layer

Persist and expose Results:

- Current local observation reads Run artifacts from the local `runs/` tree through `list_runs`, `get_run_summary` / `run-summary`, and `get_best_runs`.
- A future MLflow layer should have the Harness upload approved artifacts and metrics, with agent read-only MLflow access.

### 8. Baseline Candidate Experiments

Provide initial candidates to exercise the loop:

- single-frame UNet baseline.
- temporal stack UNet baseline.
- optional auxiliary-head UNet using `line_logits` and/or `boundary_logits`.

## First tracer-bullet milestone — complete

The first milestone was a tiny real-training vertical slice proving that the Candidate Experiment Contract and Harness can support the Research Loop. It is complete in the current codebase.

Completion evidence:

- local Candidate Experiment submission and validation are implemented;
- Run creation, copied candidate source, resolved manifests, metadata, logs, and operation outputs are implemented;
- PyTorch model smoke testing through `build_model(input_spec, output_spec)` is implemented;
- deterministic synthetic contrail-like training is implemented;
- local GVCCS-compatible data training is implemented for Single-Frame RGB Input;
- metrics, prediction samples, and local observation commands are implemented;
- Docker-backed execution and initial hardening have been added around the local loop;
- the test suite passes for the implemented tracer-bullet behavior.

The milestone remains intentionally narrower than the broader v1 contract. Temporal inputs, auxiliary heads, additional losses, pretrained-weight workflows, MLflow persistence, async orchestration, and production sandbox claims are follow-on work, not blockers for leaving the first tracer bullet.

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

   Operation-produced artifacts are written under `outputs/`; Harness-owned `candidate/`, `resolved_manifest.yaml`, and `run_metadata.json` remain at the Run root.
8. A human or agent can inspect the Run and decide what Candidate Experiment to try next.

### Explicit non-goals

The first tracer bullet does not need:

- MLflow persistence.
- asynchronous queueing.
- pretrained weight workflow.
- production security claims.
- full-scale GVCCS training beyond local GVCCS-compatible roots.

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
