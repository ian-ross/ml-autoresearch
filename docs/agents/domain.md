# Domain docs

How engineering skills should consume this repository's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repository root for canonical project vocabulary.
- Relevant ADRs under `docs/adr/`.
- Focused current docs for the area being changed, for example:
  - `docs/candidate-experiment-contract.md` for candidate validation and manifest authority;
  - `docs/run-lifecycle.md` for Run/Result behavior;
  - `docs/harness-capabilities.md` for Harness vs Research Problem ownership;
  - `docs/agent-control-boundary.md` for inner-agent authority;
  - `docs/campaign-autonomy-architecture.md` for ledger/autonomy behavior;
  - `docs/research-problem-disentangling.md` for the generic Harness/provider boundary.

This is a single-context repository. There is no required `CONTEXT-MAP.md`; if one is absent, proceed silently.

## Use the glossary's vocabulary

When your output names a domain concept, use the term as defined in `CONTEXT.md`: Harness, Candidate Experiment, Research Problem, Research Workspace Root, Run, Result, Research Ledger, Agent Control Boundary, Capability Request, Evaluation Request, and related terms.

Do not drift from the documented distinction between generic Harness behavior and provider-owned Research Problem behavior. GVCCS-specific examples belong in `docs/gvccs-features.md` temporarily and later in the GVCCS Research Problem repository.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding it. For example, changes to handoff ordering should mention ADR 0007, and changes to Research Problem packaging should mention ADRs 0008-0010.
