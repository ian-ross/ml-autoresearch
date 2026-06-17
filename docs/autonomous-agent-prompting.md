# Autonomous Agent Prompting

This document records prompting requirements for Autonomous Research Iterations.

## Agent Control Boundary prompt

When running inside the Agent Control Boundary, the agent should treat the
current directory as the writable Agent Workspace, use `ml-autoresearch-agent`
rather than `ml-autoresearch`, and hand off finalized Candidate Experiments via
`submissions/` for Harness ingestion outside the boundary. Read-only
`/reference`, `/history`, `/docs`, Research Problem Briefs, dataset profile
artifacts, and approved data summaries are context for proposal and analysis,
not places to write or sources of execution authority. Full training-data mounts
are not expected by default inside the Agent Control Boundary.

The boundary protects infrastructure authority. It is not primarily a dataset
hiding mechanism, but the agent should normally learn from Harness-owned
observations instead of untracked raw-dataset exploration. If a new class-balance
breakdown, mask statistic, subset summary, or qualitative view is needed to
answer a data-distribution question, the agent should file a Capability Request
for a Harness-generated dataset profile artifact. That request must state the
diagnostic question, expected research decision impact, scope/split, bounded
computation or artifact budget, and provenance requirements. Prefer a Candidate
Experiment when the hypothesis can be tested with current contract choices; use
an Evaluation Request when the question is about an already-completed Run.
Candidate Experiment code must remain data-path agnostic and all authoritative
Results must come from the Harness.

## Contract-bound exploration

The agent should explore model space using only the current Candidate Experiment Contract. If a desired experiment requires capability outside the contract, the agent must create a Capability Request rather than attempting to work around the Harness boundary.

Core instruction:

> If a desired experiment requires capability outside the Candidate Experiment Contract, do not emulate, tunnel, hide, or approximate that capability through candidate code. File a Capability Request instead.

## Covert workarounds are forbidden

The agent must not attempt to obtain unapproved research capability through Candidate Experiment code or helper files. Forbidden examples include:

- candidate-owned data loading or dataset path probing;
- custom training loops, losses, samplers, or transforms in helper files;
- helper modules that do anything other than define architecture layers, blocks, or model-composition code;
- runtime downloads or arbitrary checkpoint references;
- writing side-channel artifacts for unofficial evaluation;
- using model code to inspect the filesystem or environment;
- encoding multiple experiments into one Candidate Experiment to bypass Experiment Batch limits;
- disguising architecture-independent policy changes as Model Architecture code.

When blocked, the correct behavior is to write a structured Capability Request using `docs/capability-request-format.md`.
