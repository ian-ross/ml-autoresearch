# Capability Request format

A Capability Request is a human-reviewable YAML document asking the trusted Harness to expand a Harness-owned contract surface, approved resource set, operational policy, or agent-visible research context such as a dataset profile artifact.

Capability Requests are **not self-approving**. Recording a request only appends a `capability_request_created` Research Ledger event for audit. Any Candidate Experiment Contract, Approved Weight Artifact, Data Policy, execution policy, dataset profile artifact, or other Harness change must happen later through a separate human-supervised process.

## File format

Capability Request files are YAML mappings validated by the Harness-owned API/CLI. Agents should create and validate them with `ml-autoresearch-agent create-capability-request` and `ml-autoresearch-agent validate-capability-request` before finalizing a handoff, rather than hand-writing fragile YAML. Use `candidate_authority_requested: none` unless a human reviewer asks for a different description; requests should normally describe a minimal Harness change, not candidate self-permission.

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

Dataset statistic/profile requests use `capability_type: dataset_profile_artifact` and add dataset-profile details:

```yaml
request_id: capability-gvccs-mask-area-profile
capability_type: dataset_profile_artifact
blocked_hypothesis: Tiny positive masks may need a recall-oriented architecture change.
current_contract_insufficiency: Existing dataset profile artifacts do not summarize positive-mask area by split.
expected_research_value: This would show whether missed positives are dominated by small masks before proposing a new Candidate Experiment.
safety_reproducibility_risks: The summary must be generated deterministically without exposing raw training images.
minimal_harness_change: Generate a durable dataset profile artifact with mask-area histograms for train and validation splits.
candidate_authority_requested: none
example_follow_up_experiments:
  - Compare a thin-structure recall model against the current best if small masks dominate missed positives.
priority: medium
diagnostic_question: Are positive Contrail Masks concentrated in a small-area tail that explains recent false negatives?
expected_research_decision_impact: Decide whether the next Candidate Experiment should prioritize thin-structure recall or a different error mode.
scope_split: GVCCS Working Validation Split and training split; aggregate counts only.
bounded_computation_artifact_budget: One offline scan producing one YAML/Markdown summary and up to four provenance-linked plots.
provenance_requirements: Record dataset version, split definition, generation command, code version, and source mask identifiers or hashes.
```

## Required fields

- `capability_type`: one of `contract_surface`, `approved_resource`, `operational_policy`, or `dataset_profile_artifact`.
- `blocked_hypothesis`: research hypothesis blocked by the current Harness-owned surface.
- `current_contract_insufficiency`: why the current Candidate Experiment Contract, Harness policy, or exposed Research Problem context is insufficient.
- `expected_research_value`: learning value the request unlocks.
- `safety_reproducibility_risks`: risks for human reviewers.
- `minimal_harness_change`: smallest Harness-owned change that could unblock the hypothesis. For new statistics requests, describe the dataset profile artifact or summary the Harness should generate, not raw training-data access for the agent.
- `candidate_authority_requested`: one of `none`, `read_only_harness_metadata`, or `other`; prefer `none`.
- `example_follow_up_experiments`: non-empty list of Candidate Experiments that would become possible.
- `priority`: one of `low`, `medium`, `high`, or `urgent`.

For `capability_type: dataset_profile_artifact`, these additional fields are required:

- `diagnostic_question`: concrete dataset question the new statistic, subset summary, or qualitative view should answer.
- `expected_research_decision_impact`: how the artifact will change the next research decision, such as choosing a Candidate Experiment, stopping a line, or requesting another capability.
- `scope_split`: dataset scope and split(s) to summarize; avoid broad raw-data access requests.
- `bounded_computation_artifact_budget`: limits on scan cost, generated tables/figures, sample counts, or storage.
- `provenance_requirements`: dataset version, split definition, generation code/command, source identifiers/hashes, and other metadata needed to regenerate and audit the artifact.

Use a dataset-profile Capability Request when the missing information concerns the Research Problem data distribution. Use a Candidate Experiment when the next step fits the existing Candidate Experiment Contract. Use an Evaluation Request when the question concerns an already-completed Run and can be answered by approved Post-Run Evaluation over that Run's artifacts.

`request_id` is optional. If omitted, the Harness uses the request filename stem as the stable request identifier recorded in the Research Ledger.

## Agent-safe CLI

Inside the Agent Control Boundary, create Capability Request YAML from structured fields so values containing YAML metacharacters remain strings:

```bash
ml-autoresearch-agent create-capability-request \
  --output capability-requests/capability-temporal-inputs.yaml \
  --capability-type contract_surface \
  --blocked-hypothesis "Temporal context could improve thin contrail segmentation." \
  --current-contract-insufficiency "The current Candidate Experiment Contract only exposes single-frame RGB inputs." \
  --expected-research-value "This would test whether adjacent frames reduce false negatives." \
  --safety-reproducibility-risks "Temporal grouping must remain Harness-owned and deterministic." \
  --minimal-harness-change "Add an allowlisted centered temporal clip Input Mode." \
  --candidate-authority-requested none \
  --example-follow-up-experiment "temporal_candidate: compare single-frame RGB against centered temporal RGB clip." \
  --priority medium
```

Then validate before finalizing the handoff:

```bash
ml-autoresearch-agent validate-capability-request \
  --request capability-requests/capability-temporal-inputs.yaml
```

Invalid request files fail before ingestion with a JSON `status: invalid` response and an actionable schema message, including the common YAML error where an unquoted list item such as `candidate: description` parses as a mapping instead of `example_follow_up_experiments: list[str]`.

## Harness CLI

Outside the Agent Control Boundary, the Harness records validated requests:

```bash
python -m ml_autoresearch.cli create-capability-request \
  --request docs/requests/capability-temporal-inputs.yaml \
  --ledger-path research-ledger.jsonl
```

The Harness command validates the file and records a `capability_request_created` event with `request_id` and `request_path`. Invalid requests fail without appending a ledger event.
