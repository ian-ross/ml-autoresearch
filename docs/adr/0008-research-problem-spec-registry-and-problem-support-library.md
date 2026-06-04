# Research Problem Specs register problem capabilities before repository extraction

ML Autoresearch will introduce a trusted Research Problem Spec Registry as the near-term seam between reusable Harness infrastructure and Research Problem-specific policy. A Research Problem Spec is a Harness-side registration for one Research Problem's concrete capabilities: dataset adapters, input modes, prediction targets, metrics, augmentation policies, auxiliary targets, allowed losses, reporting templates, figure selectors, and similar problem-owned choices. The registry lets the Harness discover these capabilities without hard-coding every Research Problem into core infrastructure.

This creates a three-layer architecture:

1. The reusable Harness owns generic mechanisms such as Candidate Experiment validation, Run lifecycle, execution boundaries, artifact persistence, Research Ledger updates, queueing, budget policy, and other problem-independent orchestration.
2. Trusted Research Problem Specs register problem policy and concrete capabilities for each Research Problem.
3. A trusted Problem Support Library provides reusable segmentation, imaging, metric, target-construction, and reporting building blocks that Research Problem Specs and the Harness can share when multiple Research Problems need similar implementations.

Research Problem Specs and the Problem Support Library are trusted infrastructure. They are not Candidate Experiment plugins, not a candidate-facing extension point, and not authority for Candidate Experiments to bypass the Candidate Experiment Contract. Candidate Experiments continue to express variation only through Harness-owned allowlisted interfaces.

Ground-Camera Contrail Detection remains the only real registered Research Problem initially. Other Research Problems, such as future satellite contrail detection, should not drive premature generalization until the GVCCS proving-ground loop shows which seams are stable. The registry should support later extraction into trusted Research Problem Repositories, but repository extraction is a later packaging boundary rather than the first implementation step.

Temporal input should be implemented after the Research Problem Spec Registry seam, not before it. Centered temporal clip behavior depends on GVCCS-specific Frame Sequence, Target Frame, sampling, augmentation, and reporting semantics; those should be registered as Ground-Camera Contrail Detection capabilities rather than added as ad hoc core Harness behavior.
