# Research Problem Spec registry and Problem Support Library

Status: **Accepted; extended by ADR 0009 and ADR 0010.**

ML Autoresearch uses a trusted Research Problem Spec Registry as the seam between reusable Harness infrastructure and Research Problem policy. A Research Problem Spec registers one problem's capabilities: input modes, outputs, metrics, data policies, auxiliary targets, allowed losses/optimizers, brief documents, dataset profile artifacts, and training/evaluation adapters. The registry lets the Harness validate and execute candidates without hard-coding each Research Problem.

The architecture has three layers:

1. The reusable Harness owns generic mechanisms: Candidate Experiment validation, Run lifecycle, execution boundaries, artifact persistence, Research Ledger updates, autonomy handoff ingestion, and other problem-independent orchestration.
2. Trusted Research Problem Specs register policy and capabilities for each Research Problem.
3. A trusted Problem Support Library provides reusable segmentation, imaging, metric, and target-construction helpers that providers may use across Research Problems.

Research Problem Specs and the Problem Support Library are trusted infrastructure. They are not Candidate Experiment plugins and cannot let Candidate Experiments bypass the Candidate Experiment Contract.

ADR 0009 and ADR 0010 extend this decision: filesystem Research Problem packages are now the normal configured source of checked specs, and Research Problem repositories are also Research Workspace Roots. Provider specs, not ad hoc generic Harness behavior, should advertise temporal inputs and other domain-specific features.
