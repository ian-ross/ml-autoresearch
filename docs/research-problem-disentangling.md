# Research Problem disentangling plan

ML Autoresearch is moving toward a reusable Harness package plus trusted filesystem Research Problem packages. The initial Research Problem package will be GVCCS / Ground-Camera Contrail Detection, but the Harness should not need to know GVCCS-specific concepts in order to validate, run, evaluate, or record Candidate Experiments for other Research Problems.

See also:

- `docs/adr/0006-trusted-research-problem-repositories-register-problem-capabilities.md`
- `docs/adr/0008-research-problem-spec-registry-and-problem-support-library.md`
- `docs/adr/0009-filesystem-research-problem-packages-provide-checked-specs.md`

## End state

The reusable Harness owns problem-independent mechanisms:

- Candidate Experiment validation flow;
- Run lifecycle and metadata persistence;
- Candidate Execution Boundary and execution backend policy;
- Research Ledger updates;
- queueing, budgets, retries, and autonomous-step orchestration;
- generic artifact locations and provenance recording.

A trusted Research Problem package owns Research Problem policy and concrete capabilities:

- dataset discovery, validation, metadata, and loading;
- split and frame-selection semantics;
- input mode realization;
- prediction target and auxiliary target derivation;
- candidate-selectable Data Policy choices that depend on the Research Problem;
- loss and metric choices scoped to that Research Problem;
- qualitative prediction sample selection and figure/reporting semantics;
- Post-Run Evaluation modes that depend on the data modality and prediction target.

The Harness accesses this code by loading a configured filesystem Research Problem package provider, validating the returned Research Problem Spec, registering it, and then using only the checked Spec interface and adapters.

## Current GVCCS leaks in reusable Harness modules

The present codebase already has a Research Problem Spec Registry and a Problem Support Library, but many reusable Harness modules still import or name GVCCS directly:

- `src/ml_autoresearch/gvccs.py` contains the trusted GVCCS dataset adapter and temporal-frame semantics, but it currently lives inside the Harness package.
- `src/ml_autoresearch/training.py` directly discovers GVCCS samples, builds GVCCS datasets, applies GVCCS-specific augmentation policy, derives Contrail Mask auxiliary targets, and selects `val/dice`.
- `src/ml_autoresearch/runs.py`, `src/ml_autoresearch/batches.py`, `src/ml_autoresearch/execution.py`, `src/ml_autoresearch/container_runner.py`, and `src/ml_autoresearch/cli.py` expose `gvccs`-named training operations.
- `src/ml_autoresearch/evaluations.py` and `src/ml_autoresearch/evaluation_requests.py` directly run GVCCS whole-validation failure analysis.
- `src/ml_autoresearch/artifacts.py` uses `GVCCSSample`, `Frame Sequence`, positive Contrail Mask assumptions, and channel-stacked temporal RGB rendering.
- `src/ml_autoresearch/autonomy_step.py` infers GVCCS data roots and dispatches GVCCS-specific run and batch execution.
- tests and fixture candidates use GVCCS terms heavily; those can remain as Research Problem-specific tests, but reusable Harness tests need a fake Research Problem package to prove the seam.

## Target module ownership

Near-term, external Research Problem package code should be placed in pre-initialised local Git repositories mounted into the development VM:

- `/home/iross/code/test-research-problem` for any fake or external test Research Problem package needed to prove the Harness seam;
- `/home/iross/code/gvccs-research-problem` for the eventual GVCCS / Ground-Camera Contrail Detection Research Problem package.

These repositories should behave as separate filesystem Research Problem packages even when implementation work is coordinated with this Harness repository.

Suggested ownership:

- Harness package: `ml_autoresearch`
  - generic Research Problem provider loader;
  - Research Problem Spec Registry and contract types;
  - generic Run, batch, execution, evaluation, and autonomy dispatch;
  - generic Candidate Experiment Contract validation using registered specs;
  - Problem Support Library helpers that are reusable across multiple Research Problems.
- GVCCS package: `gvccs`
  - `gvccs.research_problem:build_spec`;
  - GVCCS dataset adapter, temporal clip adapter, split/frame-selection policy;
  - Ground-Camera Contrail Detection input/output/metric/augmentation/auxiliary-target policy;
  - GVCCS prediction sample and Post-Run Evaluation adapters.

## Vertical migration slices

1. Add filesystem Research Problem provider loading and provenance recording using a tiny fake Research Problem package in tests.
2. Move the built-in Ground-Camera Contrail Detection Spec behind a provider function and make Candidate validation use the loaded provider instead of a hard-coded default.
3. Replace `train_gvccs*` APIs with a generic Research Problem training dispatch that asks the Spec for data loaders, metric selection, and target/loss behavior.
4. Move GVCCS dataset and input-mode construction out of reusable Harness imports and into the GVCCS package.
5. Move augmentation policy and auxiliary target derivation behind Research Problem adapters while keeping reusable segmentation helpers in the Problem Support Library.
6. Replace GVCCS-specific prediction sample selection in the generic artifact writer with a Research Problem figure/sample selector adapter.
7. Replace direct GVCCS Post-Run Evaluation imports with Research Problem evaluation adapter dispatch.
8. Generalize CLI, batch execution, container runner commands, and autonomous-step execution so they select a configured Research Problem instead of calling GVCCS-specific entrypoints.
9. Add regression tests that the reusable Harness imports and core tests pass without importing the GVCCS package, except through explicit configured provider loading.

## Deletion test

After disentangling, temporarily removing the GVCCS Research Problem package should leave the reusable Harness importable and its problem-independent tests runnable. Operations configured for Ground-Camera Contrail Detection should fail clearly at Research Problem provider loading or registration, not because core Harness modules import `ml_autoresearch.gvccs` directly.
