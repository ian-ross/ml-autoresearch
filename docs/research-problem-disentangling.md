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

## Completed disentangling status

The migration slices are complete for the current tracer-bullet Harness. Reusable Harness modules no longer contain the GVCCS implementation package and no longer directly import GVCCS dataset types or adapters during package import. Research Problem-specific behavior is reached through checked Spec adapters loaded from filesystem provider paths, and regression tests now guard that reusable Harness modules do not add new direct GVCCS imports.

Canonical filesystem packages:

- `/home/iross/code/test-research-problem` contains the external fake Research Problem package used by deletion/seam regression tests.
- `/home/iross/code/gvccs-research-problem` contains the GVCCS / Ground-Camera Contrail Detection package. Its provider target is `gvccs.research_problem:build_spec`.

Intentional remaining exceptions:

- `train_gvccs`, `run_candidate_with_gvccs_data`, and `run_experiment_batch_with_gvccs_data` are legacy compatibility wrappers around generic Research Problem dispatch for older GVCCS workflows. They are not the primary reusable Harness API.
- GVCCS-specific tests, fixtures, research notes, Candidate Experiments, and campaign artifacts intentionally keep GVCCS terminology because they describe the initial Research Problem history.

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

All nine planned slices are complete for the current Harness boundary:

1. Filesystem Research Problem provider loading and provenance recording were added with fake-package tests.
2. The built-in Ground-Camera Contrail Detection Spec moved behind a provider function and Candidate validation can use loaded provider registries.
3. Generic Research Problem training dispatch asks the Spec adapter for datasets, metrics, losses, targets, and data-policy behavior.
4. GVCCS dataset and input-mode construction live behind the GVCCS Research Problem adapter, not reusable Harness imports.
5. Augmentation policy and auxiliary target derivation are adapter-owned, with reusable segmentation helpers kept in the Problem Support Library.
6. Prediction sample selection and figure rendering dispatch through adapter hooks.
7. Post-Run Evaluation dispatch uses the Research Problem evaluation adapter.
8. CLI, batch execution, container runner, and autonomous-step flows have generic Research Problem paths; GVCCS-named paths are compatibility-only wrappers.
9. Regression tests prove reusable Harness imports and a fake Research Problem flow work while the GVCCS package is simulated as unavailable.

## Deletion test

After disentangling, temporarily removing the GVCCS Research Problem package should leave the reusable Harness importable and its problem-independent tests runnable. Operations configured for Ground-Camera Contrail Detection should fail clearly at Research Problem provider loading or registration, not because core Harness modules import `ml_autoresearch.gvccs` directly.
