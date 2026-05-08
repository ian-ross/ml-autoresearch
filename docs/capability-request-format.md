# Capability Request Format

A Capability Request is a structured, human-gated request from the agent to expand Harness-owned contract surface, approved resources, or operational policy when current capabilities block useful Research Progress.

Use this format when the current Candidate Experiment Contract prevents a useful next experiment.

```yaml
capability_type: input_mode | output_form | loss | augmentation_policy | sampling_policy | metric | approved_weight | budget_policy | reporting | other
blocked_hypothesis: "..."
why_current_contract_insufficient: "..."
expected_research_value: "..."
safety_or_reproducibility_risks: "..."
minimal_harness_change: "..."
candidate_authority_requested: "none"
example_followup_experiments:
  - "..."
priority: low | medium | high
```

Rules:

- Capability Requests are not self-approving.
- The agent may continue using existing Candidate Experiment Contract choices while a Capability Request is pending.
- Candidate authority requested should normally be `none`; prefer Harness-owned parameters or approved resources.
- A request should explain the blocked hypothesis, not merely ask for a feature.
