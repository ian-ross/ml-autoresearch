# Autonomous Agent Prompting

This document records prompting requirements for Autonomous Research Iterations.

## Agent Control Boundary prompt

Inside the Agent Control Boundary, the agent should treat the current directory
as the writable Agent Workspace, use `ml-autoresearch-agent` rather than
`ml-autoresearch`, and hand off finalized Candidate Experiments via
`submissions/` for Harness ingestion outside the boundary. Read-only
`/reference`, `/history`, `/docs`, Research Problem Briefs, dataset profile
artifacts, and approved data summaries are proposal and analysis context, not
write targets or execution authority. Full training-data mounts are not expected
by default.

The boundary protects infrastructure authority. It is not primarily a dataset
hiding mechanism, but the agent should normally learn from Harness-owned
observations instead of untracked raw-dataset exploration. If a new class-balance
breakdown, mask statistic, subset summary, or qualitative view is needed for a
data-distribution question, the agent should file a Capability Request for a
Harness-generated dataset profile artifact. The request must state the
diagnostic question, expected research impact, scope/split, bounded computation
or artifact budget, and provenance requirements. Prefer a Candidate Experiment
when current contract choices can test the hypothesis; use an Evaluation Request
for questions about an already-completed Run. Candidate Experiment code must
remain data-path agnostic, and all authoritative Results must come from the
Harness.

## Contract-bound exploration

The agent should explore model space only within the current Candidate Experiment Contract. If a desired experiment requires capability outside the contract, the agent must create a Capability Request rather than work around the Harness boundary.

Core instruction:

> If a desired experiment requires capability outside the Candidate Experiment Contract, do not emulate, tunnel, hide, or approximate that capability through candidate code. File a Capability Request instead.

## Architecture family exploration policy

Use the current best Result as the promotion target, not as the only criterion for whether an immature architecture family deserves further development. A substantially new family's first successful Run is a scouting Run unless the proposal explicitly defines it as a mature promotion candidate. Do not abandon a new family solely because one untuned or lightly tuned scout fails to beat a heavily tuned incumbent.

Separate three decisions:

- **Promotion candidate:** must beat or credibly tie the current best Result on the Research Problem selection metric and required guardrails.
- **Family scout:** does not need to beat the current best; it should test whether the family has a plausible strength, such as competitive metric distance, improved secondary metrics, better failure buckets, lower resource cost, useful learning dynamics, or qualitatively different errors.
- **Family-development continuation:** should compare against the family baseline and prior scouts, with a bounded sequence of controlled variants before declaring the family exhausted.

Before rejecting a substantially new architecture family, either run a bounded family-development sequence with comparable tuning depth to the incumbent lineage, or document a hard stop reason: Candidate Experiment Contract violation, parameter/resource infeasibility, catastrophic metric gap, unstable training, implementation defect, or diagnostics showing that the family cannot address the campaign failure modes. Compare mature families to mature incumbents; compare scouts to scouts when allocating research budget.

## Covert workarounds are forbidden

The agent must not obtain unapproved research capability through Candidate Experiment code or helper files. Forbidden examples include:

- candidate-owned data loading or dataset path probing;
- custom training loops, losses, samplers, or transforms in helper files;
- helper modules that do anything other than define architecture layers, blocks, or model-composition code;
- runtime downloads or arbitrary checkpoint references;
- writing side-channel artifacts for unofficial evaluation;
- using model code to inspect the filesystem or environment;
- encoding multiple experiments into one Candidate Experiment to bypass Experiment Batch limits;
- disguising architecture-independent policy changes as Model Architecture code.

When blocked, the correct behavior is to write a structured Capability Request using `docs/capability-request-format.md`.
