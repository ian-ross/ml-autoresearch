# Candidate-selectable Data Policy

Candidate Experiments may select trusted allowlisted Data Policy choices, including Sampling Policy, Frame Selection Policy, and Augmentation Policy, without providing custom data loaders, samplers, transforms, dataset paths, or training loops. Allowed values come from the active Research Problem Spec and trusted provider code.

This preserves Harness ownership of data loading while allowing useful research variation: the security risk is arbitrary code or filesystem authority, not choosing among audited implementations.
