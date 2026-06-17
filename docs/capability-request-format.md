# Capability Request format

A Capability Request is a human-reviewable YAML document for asking that the trusted Harness expand a Harness-owned contract surface, approved resource set, operational policy, or agent-visible research context such as a dataset profile artifact.

Capability Requests are **not self-approving**. Recording a request only appends a `capability_request_created` Research Ledger event for auditability. Any Candidate Experiment Contract, Approved Weight Artifact, Data Policy, execution policy, dataset profile artifact, or other Harness change must happen later through a separate human-supervised process.

## File format

Capability Request files are YAML mappings validated by the Harness-owned API/CLI. Use `candidate_authority_requested: none` unless a human reviewer explicitly asks for a different description; requests should normally describe a minimal Harness change, not candidate self-permission.

```yaml
request_id: capability-temporal-inputs
capability_type: contract_surface
blocked_hypothesis: Temporal context could improve thin contrail segmentation.
current_contract_insufficiency: The current Candidate Experiment Contract only exposes single-frame RGB inputs.
expected_research_value: This would test whether adjacent frames reduce false negatives.
safety_reproducibility_risks: Temporal grouping must remain Harness-owned and deterministic.
minimal_harness_change: Add an allowlisted centered temporal clip Input Mode.
candidate_authority_requested: none
example_follow_up_experiments:
  - Compare single-frame RGB against centered temporal RGB clip.
priority: medium
```

## Required fields

- `capability_type`: one of `contract_surface`, `approved_resource`, `operational_policy`, or `dataset_profile_artifact`.
- `blocked_hypothesis`: the research hypothesis blocked by the current Harness-owned surface.
- `current_contract_insufficiency`: why the current Candidate Experiment Contract, Harness policy, or exposed Research Problem context is insufficient.
- `expected_research_value`: what learning value the request would unlock.
- `safety_reproducibility_risks`: risks that human reviewers must consider.
- `minimal_harness_change`: smallest Harness-owned change that could unblock the hypothesis. For new statistics requests, describe the dataset profile artifact or summary the Harness should generate, not raw training-data access for the agent.
- `candidate_authority_requested`: one of `none`, `read_only_harness_metadata`, or `other`; prefer `none`.
- `example_follow_up_experiments`: non-empty list of Candidate Experiments that would become possible.
- `priority`: one of `low`, `medium`, `high`, or `urgent`.

`request_id` is optional. If omitted, the Harness uses the request filename stem as the stable request identifier recorded in the Research Ledger.

## CLI

```bash
python -m ml_autoresearch.cli create-capability-request \
  --request docs/requests/capability-temporal-inputs.yaml \
  --ledger-path research-ledger.jsonl
```

The command validates the file and records a `capability_request_created` event containing `request_id` and `request_path`. Invalid requests fail without appending a ledger event.
