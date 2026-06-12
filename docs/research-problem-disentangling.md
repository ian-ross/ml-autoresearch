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

## Research Problem Brief documents

The Research Problem Spec is the normative, machine-checkable execution contract: it declares the input modes, output forms, losses, optimizers, metrics, data-policy choices, and trusted adapters that the Harness validates and uses during Runs. Missing required contract fields remain hard errors.

A filesystem Research Problem package may also declare Research Problem Brief documents through `brief_documents` on `ResearchProblemSpec`. Each brief document has a stable `name`, a stable `role` (for example `problem_overview`, `domain_data_notes`, `literature_references`, `baseline_description`, or `modeling_suggestions`), a package-relative `path`, an optional short `summary`, and an optional `required` flag.

Brief documents are advisory context for progressive disclosure to humans and agents. The Harness resolves their paths relative to the Research Problem package root and exposes the resolved metadata after provider loading, but it does not treat advisory missing files as execution-contract failures. A document marked `required=True` must exist. Absolute paths, parent-directory escapes such as `../notes.md`, backslash-separated paths, and paths that resolve outside the package root are rejected so a provider cannot accidentally point into the Harness repository or another filesystem location.

## Disentangling status

The current boundary is that production code in the reusable `ml_autoresearch` package is Research Problem-generic. It must not contain a GVCCS implementation package, directly import GVCCS dataset types or adapters, hard-code GVCCS provider targets, expose GVCCS-named production APIs/commands, or mention GVCCS in production defaults or error messages. Research Problem-specific behavior is reached through checked Spec adapters loaded from configured filesystem provider paths, and regression tests scan tracked `src/ml_autoresearch/**` files to prevent GVCCS-specific production references from returning.

Canonical filesystem packages:

- The external fake Research Problem package is used by deletion/seam regression tests. Local development may use `/home/iross/code/test-research-problem`; CI should check out or create the package in a scratch location and pass that root to tests through configuration such as `ML_AUTORESEARCH_TEST_PROBLEM_ROOT`.
- The GVCCS / Ground-Camera Contrail Detection package is an example filesystem Research Problem package that provides target `gvccs.research_problem:build_spec`. Local development may use `/home/iross/code/gvccs-research-problem`; CI should check it out in a scratch location and pass that root to tests through configuration such as `ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT`.
- Tests should not depend on sibling checkout layout or user-specific absolute paths; when testing the filesystem package seam, they should assert that packages are outside the reusable `src/ml_autoresearch` package rather than at a specific absolute location.
- External-package integration tests are required because filesystem Research Problem packages are a core system boundary. If the configured package roots are missing or invalid, those tests should fail clearly rather than skip by default.

### Test runner configuration

Set the external package roots explicitly before running integration-focused tests:

```bash
export ML_AUTORESEARCH_TEST_PROBLEM_ROOT=/path/to/test-research-problem
export ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT=/path/to/gvccs-research-problem

uv run pytest tests/test_research_problem_provider_loader.py tests/test_research_problem_disentangling.py tests/test_gvccs_data.py
```

CI should inject the same variables when checking out the companion package repositories.

Intentional remaining exceptions:

- GVCCS-specific tests, fixtures, research notes, Candidate Experiments, and campaign artifacts may keep GVCCS terminology because they describe the initial Research Problem history.
- Production code in the reusable `ml_autoresearch` package must not expose GVCCS-named APIs, commands, compatibility wrappers, defaults, error messages, provider targets, or filesystem paths; it uses generic Research Problem provider configuration and dispatch.

## Target module ownership

Near-term, external Research Problem package code should be placed in pre-initialised local Git repositories mounted into the development VM:

- `/home/iross/code/test-research-problem` for any fake or external test Research Problem package needed to prove the Harness seam;
- `/home/iross/code/gvccs-research-problem` for the GVCCS / Ground-Camera Contrail Detection example Research Problem package.

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

The remaining migration should complete these planned slices for the current Harness boundary:

1. Filesystem Research Problem provider loading and provenance recording were added with fake-package tests.
2. Candidate validation can use Research Problem Specs loaded from configured filesystem providers.
3. Generic Research Problem training dispatch asks the Spec adapter for datasets, metrics, losses, targets, and data-policy behavior.
4. GVCCS dataset and input-mode construction live behind the GVCCS Research Problem adapter, not reusable Harness imports.
5. Augmentation policy and auxiliary target derivation are adapter-owned, with reusable segmentation helpers kept in the Problem Support Library.
6. Prediction sample selection and figure rendering dispatch through adapter hooks.
7. Post-Run Evaluation dispatch uses the Research Problem evaluation adapter.
8. CLI, batch execution, container runner, and autonomous-step flows have generic Research Problem paths and no GVCCS-named production wrappers.
9. Regression tests prove reusable Harness imports and a fake Research Problem flow work while the GVCCS package is simulated as unavailable, and fail if GVCCS-specific production paths are reintroduced.

## Deletion test

After disentangling, temporarily removing the GVCCS Research Problem package should leave the reusable Harness importable and its problem-independent tests runnable. Operations configured for Ground-Camera Contrail Detection should fail clearly at Research Problem provider loading or registration, not because core Harness modules import `ml_autoresearch.gvccs` directly.
