# Autonomy Steps ingest one agent handoff before execution

An Autonomy Step is the outer-Harness orchestration unit for building Autonomous Research Iterations incrementally: it refreshes the Agent Control Boundary, invokes or supports Pi once with a Harness-generated one-step prompt, ingests exactly one primary agent handoff, records canonical state, and selects at most one next action.

The agent must do one thing at a time. If blocked, it writes a Campaign Report rather than mixing a Capability Request, Evaluation Request, Candidate Submission, Research Note, Experiment Batch Submission, or Campaign Report in one step.

We separate Agent Handoff Ingestion from Candidate Experiment execution because agent artifacts are research actions needing their own audit trail, while Runs and Post-Run Evaluations remain Harness-owned execution actions. Ingestion validates before copying, copies rather than moves, rejects duplicate canonical destinations, updates canonical docs such as `EXPERIMENT_INDEX.md` when needed, records `agent_handoff_ingested` plus applicable semantic events, and writes the source-side ingestion marker last.

Candidate, evaluation, and batch execution happens only after ingestion when explicitly enabled. `run-autonomous-iteration` is a bounded loop over the same Autonomy Step invariants, not a license for unbounded autonomous continuation.
