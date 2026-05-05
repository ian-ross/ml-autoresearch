# Candidate-selectable Data Policy

Candidate Experiments may select from Harness-owned allowlisted Data Policy choices, including Sampling Policy and Augmentation Policy, without providing custom data loaders, samplers, transforms, dataset paths, or training loops. This preserves the Harness ownership established for data loading while allowing useful research variation where the security risk comes from arbitrary code or filesystem authority, not from selecting among audited Harness implementations.
