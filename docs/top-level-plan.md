# ML Autoresearch status and roadmap

This document is the current high-level status and roadmap for the reusable ML Autoresearch Harness. `CONTEXT.md` defines the project language; ADRs record decision history; detailed contracts live in the focused docs linked below.

## Current architecture

ML Autoresearch is a reusable Harness package plus one or more trusted filesystem Research Problem packages.

- The **Harness** owns validation, execution policy, run/evaluation artifact layout, Research Ledger writes, runtime image workflow, Agent Control Boundary preparation, and autonomous handoff ingestion.
- A **Research Problem package** owns domain semantics: input modes, output specs, allowed losses/optimizers/data policies, dataset adapters, training/evaluation adapters, brief documents, and dataset profile artifacts.
- Candidate Experiments are untrusted research packages constrained by `docs/candidate-experiment-contract.md`; candidate code does not choose data paths, training loops, mounts, network access, Docker, or ledger writes.
- A Research Workspace Root contains `ml-autoresearch.toml`, the canonical Research Ledger, candidates, runs, batches, notes, handoff directories, and generated agent/workspace state.

## Implemented workflows

Current implemented Harness workflows include:

1. `setup` creates a Research Workspace Root skeleton and configuration.
2. `submit-candidate` performs static source/manifest/proposal validation and records/copies accepted submissions.
3. `run-candidate` validates, smoke-tests, and trains a candidate through the configured Research Problem provider, using Docker by default for execution.
4. `evaluate-run` and request-gated `run-post-run-evaluation` produce run-scoped evaluation artifacts.
5. Experiment Batches validate a small batch atomically before executing sibling runs independently.
6. Campaign/report/capability/evaluation events are written to the append-only `research-ledger.jsonl` through Harness-owned APIs.
7. `prepare-agent-boundary`, `autonomy-step`, and `run-autonomous-iteration` prepare bounded agent context, ingest exactly one handoff, and optionally execute the next Harness-owned action.
8. Runtime image commands stage, build, validate, and stamp workspace-specific Agent Runtime Image assets and Docker runner image tags.

## Current generic contracts

- Candidate contract: `docs/candidate-experiment-contract.md`
- Run lifecycle: `docs/run-lifecycle.md`
- Harness capabilities and ownership: `docs/harness-capabilities.md`
- Campaign/autonomy architecture: `docs/campaign-autonomy-architecture.md`
- Agent Control Boundary: `docs/agent-control-boundary.md`
- Dependency/runtime image strategy: `docs/dependency-strategy.md`
- Experiment batches: `docs/experiment-batches.md`
- Request/report formats:
  - `docs/capability-request-format.md`
  - `docs/evaluation-request-format.md`
  - `docs/campaign-report-format.md`

## Temporary GVCCS notes

Ground-Camera Contrail Detection / GVCCS is the initial Research Problem package used to exercise the generic Harness seam. GVCCS-specific data layout, input modes, auxiliary targets, and candidate families are collected temporarily in `docs/gvccs-features.md`; those notes should move to the GVCCS Research Problem repository when that repository becomes the sole home for problem-specific documentation.

## Follow-on roadmap

Follow-on work should be described as future capability unless tests and code implement it. Current roadmap candidates include:

- Approved Weight Artifacts and Pretrained Weight Requests. Current code forbids candidate-supplied checkpoints and arbitrary runtime downloads; it does not yet implement an approved-weight registry/workflow.
- Additional provider or Harness-supported candidate knobs such as extra losses, optimizers, augmentation DSLs, mixed precision controls, gradient clipping, and advanced sampling policies.
- Async scheduling and richer production isolation beyond the current Docker-backed Candidate Execution Boundary.
- Additional Research Problem packages that prove the provider seam is not GVCCS-specific.

## Non-goals for the current Harness

- Candidate-owned dataset loading, arbitrary filesystem traversal, custom training loops, runtime weight downloads, Docker invocation, or network authority.
- Generic Harness production code that imports or hard-codes GVCCS-specific types, provider targets, paths, or commands.
- Treating Agent Control Boundary work as authoritative training/evaluation. Authoritative Results come from Harness-owned execution outside the inner agent VM.
