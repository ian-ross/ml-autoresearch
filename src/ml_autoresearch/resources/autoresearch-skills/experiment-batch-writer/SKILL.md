---
name: experiment-batch-writer
description: Prepare one small, controlled Experiment Batch Submission under Harness-owned resource policy.
---

# Experiment Batch Writer

## Use

Use only when two to four related Candidate Experiments test one shared hypothesis or a small controlled comparison. A batch is one primary Autonomy Step handoff, not an arbitrary sweep.

## Read first

- `AGENTS.md` for the current Harness-owned batch size, parallel Run cap, and resource-profile policy.
- `docs/experiment-batches.md` for batch structure, validation, execution, and artifacts.
- `docs/candidate-experiment-contract.md` for Candidate boundaries.
- `../proposal-writer/SKILL.md` and `../candidate-implementer/SKILL.md`.

## Instructions

1. Confirm every proposed candidate belongs to one shared hypothesis and differs by a documented controlled factor.
2. Confirm the architecture family and requested batch sizes have trusted resource evidence. Unprofiled architecture families must be submitted sequentially.
3. Create one draft directory containing `BATCH_PROPOSAL.md` and `candidates/<candidate_id>/` entries.
4. In `BATCH_PROPOSAL.md`, record the shared hypothesis, comparison target, per-candidate rationale, decision criteria, success criteria, requested budget, and requested concurrency subject to the lower Harness cap in `AGENTS.md`.
5. Keep every candidate architecture-only and select only trusted manifest policies.
6. Prepare exactly one immutable handoff:

```bash
ml-autoresearch-agent prepare-experiment-batch-submission \
  --batch <draft-batch-directory> \
  --submissions-root batch-submissions
```

7. Stop after preparation. The Harness and human review own ingestion, GPU placement, concurrency, and execution.

## Guardrails

- Never use a batch to bypass the one-primary-handoff rule, Candidate validation, budget review, or candidate-count limits.
- Never put unrelated architecture scouts into one batch merely to increase throughput.
- Never implement GPU selection, process launching, training loops, resource measurement, or retry policy in Candidate code.
- Never assume parameter count predicts activation memory; use trusted measured resource profiles.
- If safe concurrency is unknown or candidates have materially different resource envelopes, submit one Candidate Experiment or create a Capability Request.
