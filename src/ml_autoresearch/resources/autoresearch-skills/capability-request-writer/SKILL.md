---
name: capability-request-writer
description: Create human-reviewable Capability Requests.
---

# Capability Request Writer

## Use

Use when a hypothesis is blocked by the current Candidate Experiment Contract, Approved Weight Artifact set, Data Policy, operational policy, or missing Harness-generated dataset profile/statistic. A Capability Request is not self-approving.

## Read first

- `CONTEXT.md` for Candidate Experiment Contract, Harness, and Approved Weight Artifact terms.
- `docs/capability-request-format.md` for required fields and CLI.
- ADR-0001, ADR-0002, and ADR-0003.

## Instructions

Describe the blocked hypothesis, current insufficiency, expected research value, safety/reproducibility risks, minimal Harness-owned change, and example follow-up experiments. Prefer `candidate_authority_requested: none`. Use `ml-autoresearch-agent create-capability-request` to generate the YAML from structured options, and always run `ml-autoresearch-agent validate-capability-request --request PATH` before treating the Capability Request as final. This avoids YAML pitfalls such as an unquoted `candidate: description` follow-up parsing as a mapping instead of a string. Recording the request creates only an auditable event; it does not authorize implementation.

For dataset-statistic requests, use `capability_type: dataset_profile_artifact` and ask for a durable Harness-generated profile artifact or summary with provenance rather than raw training-data access inside the Agent Control Boundary. Include the diagnostic question, expected research decision impact, scope/split, bounded computation or artifact budget, and provenance requirements. Choose this only when the missing information concerns the Research Problem data distribution itself; propose a Candidate Experiment when the next hypothesis fits the existing contract, and write an Evaluation Request when the question concerns an already-completed Run.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
