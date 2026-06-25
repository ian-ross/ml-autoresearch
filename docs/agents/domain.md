# Domain docs

How engineering skills consume this repository's domain documentation while exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repository root for canonical project vocabulary.
- Relevant ADRs under `docs/adr/`.
- Focused current docs for the changed area, for example:
  - `docs/candidate-experiment-contract.md` for candidate validation and manifest authority;
  - `docs/run-lifecycle.md` for Run/Result behavior;
  - `docs/harness-capabilities.md` for Harness vs Research Problem ownership;
  - `docs/agent-control-boundary.md` for inner-agent authority;
  - `docs/campaign-autonomy-architecture.md` for ledger/autonomy behavior;
  - `docs/research-problem-disentangling.md` for the generic Harness/provider boundary.

This is a single-context repository. There is no required `CONTEXT-MAP.md`; if absent, proceed silently.

## Use the glossary's vocabulary

When naming a domain concept, use the term as defined in `CONTEXT.md`: Harness, Candidate Experiment, Research Problem, Research Workspace Root, Run, Result, Research Ledger, Agent Control Boundary, Capability Request, Evaluation Request, and related terms.

Keep the documented distinction between generic Harness behavior and provider-owned Research Problem behavior. GVCCS-specific examples belong temporarily in `docs/gvccs-features.md` and later in the GVCCS Research Problem repository.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly instead of silently overriding it. For example, handoff ordering changes should mention ADR 0007, and Research Problem packaging changes should mention ADRs 0008-0010.
