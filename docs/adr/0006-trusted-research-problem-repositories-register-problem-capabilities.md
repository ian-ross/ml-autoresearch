# Trusted Research Problem repositories register problem capabilities

Status: **Superseded by ADR 0008, ADR 0009, and ADR 0010.**

This ADR recorded the decision direction that reusable Harness infrastructure should be separated from Research Problem-specific definitions through trusted Research Problem repositories. That direction remains valid, but the current architecture is specified more precisely by:

- ADR 0008: Research Problem Specs and the Problem Support Library;
- ADR 0009: filesystem Research Problem packages that provide checked specs;
- ADR 0010: Research Problem repositories as Research Workspace Roots.

The retained historical decision is that problem-specific semantics such as dataset adapters, input modes, metrics, augmentation policies, auxiliary targets, losses, reporting templates, and figure selectors belong in trusted Research Problem infrastructure, not in Candidate Experiment plugins and not as hard-coded generic Harness assumptions.
