---
name: capability-request-writer
description: Create human-reviewable Capability Requests.
---

# Capability Request Writer

## Use

Use when a hypothesis is blocked by the current Candidate Experiment Contract, Approved Weight Artifact set, Data Policy, or operational policy. A Capability Request is not self-approving.

## Read first

- `CONTEXT.md` for Candidate Experiment Contract, Harness, and Approved Weight Artifact terms.
- `docs/capability-request-format.md` for required fields and CLI.
- ADR-0001, ADR-0002, and ADR-0003.

## Instructions

Describe the blocked hypothesis, current insufficiency, expected research value, safety/reproducibility risks, minimal Harness-owned change, and example follow-up experiments. Prefer `candidate_authority_requested: none`. Recording the request only creates an auditable event; it does not authorize implementation.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; Harness changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use documented run, candidate, note, request, report, and artifact paths only.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
