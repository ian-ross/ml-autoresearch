# Research Problem disentangling boundary

ML Autoresearch is a reusable Harness package plus trusted filesystem Research Problem packages. The Harness must not need GVCCS-specific concepts to validate, run, evaluate, or record Candidate Experiments for other Research Problems.

See also ADRs 0006, 0008, 0009, and 0010.

## Current boundary

The reusable Harness owns problem-independent mechanisms:

- Candidate Experiment validation flow;
- Run lifecycle and metadata persistence;
- Candidate Execution Boundary and execution backend policy;
- Research Ledger updates;
- batch, retry, budget, and autonomous-step orchestration;
- generic artifact locations and provenance recording.

A trusted Research Problem package owns Research Problem policy and concrete capabilities:

- dataset discovery, validation, metadata, and loading;
- split and frame-selection semantics;
- input mode realization;
- prediction target and auxiliary target derivation;
- candidate-selectable data-policy choices that depend on the Research Problem;
- loss, optimizer, and metric choices scoped to that Research Problem;
- qualitative prediction sample selection and reporting semantics;
- Post-Run Evaluation adapter behavior that depends on the data modality and prediction target.

The Harness accesses this code by loading a configured filesystem Research Problem provider, validating the returned Research Problem Spec, registering it, and using only the checked Spec interface and adapters.

## Research Problem Brief documents

The Research Problem Spec is the normative, machine-checkable execution contract. It declares input modes, output forms, losses, optimizers, metrics, data-policy choices, brief documents, dataset profile artifacts, and trusted adapters that the Harness validates and uses during Runs.

Brief documents are advisory context for progressive disclosure to humans and agents. The Harness resolves provider-declared brief paths relative to the Research Problem package root. Required brief files must exist; optional ones may be omitted. Absolute paths, parent-directory escapes, backslash-separated paths, and paths resolving outside the package root are rejected.

## GVCCS deletion seam

Production code in reusable `ml_autoresearch` must remain Research Problem-generic. It must not contain a GVCCS implementation package, directly import GVCCS dataset types or adapters, hard-code GVCCS provider targets, expose GVCCS-named production APIs/commands, or mention GVCCS in production defaults or error messages.

Intentional exceptions are tests, fixtures, research notes, Candidate Experiments, campaign artifacts, and temporary documentation such as `docs/gvccs-features.md`, because they describe the initial Research Problem history.

Regression tests scan tracked `src/ml_autoresearch/**` files to prevent GVCCS-specific production references from returning. Additional tests prove reusable Harness imports and fake Research Problem flows work while simulating the GVCCS package as unavailable.

## External package tests

Canonical filesystem packages for integration-focused tests are external to `src/ml_autoresearch`:

- a fake/test Research Problem package, configured with `ML_AUTORESEARCH_TEST_PROBLEM_ROOT`;
- the GVCCS / Ground-Camera Contrail Detection package, configured with `ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT` and provider target `gvccs.research_problem:build_spec`.

Tests should not depend on sibling checkout layout or user-specific absolute paths; they should assert that packages are outside the reusable Harness package. External-package integration tests are required because filesystem Research Problem packages are a core system boundary. If configured package roots are missing or invalid, those tests fail clearly rather than skip by default.

Example:

```bash
export ML_AUTORESEARCH_TEST_PROBLEM_ROOT=/path/to/test-research-problem
export ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT=/path/to/gvccs-research-problem

uv run pytest tests/test_research_problem_provider_loader.py tests/test_research_problem_disentangling.py tests/test_gvccs_data.py
```

## Module ownership

Harness package `ml_autoresearch` owns:

- generic Research Problem provider loading;
- Research Problem Spec Registry and contract types;
- generic Run, batch, execution, evaluation, and autonomy dispatch;
- generic Candidate Experiment Contract validation using registered specs;
- Problem Support Library helpers reusable across multiple Research Problems.

A GVCCS package owns:

- `gvccs.research_problem:build_spec`;
- GVCCS dataset adapter, temporal clip adapter, split/frame-selection policy;
- Ground-Camera Contrail Detection input/output/metric/augmentation/auxiliary-target policy;
- GVCCS prediction sample and Post-Run Evaluation adapters.

## Deletion test

Temporarily removing the GVCCS Research Problem package should leave the reusable Harness importable and its problem-independent tests runnable. Operations configured for Ground-Camera Contrail Detection should fail clearly at Research Problem provider loading or registration, not because core Harness modules import GVCCS directly.
