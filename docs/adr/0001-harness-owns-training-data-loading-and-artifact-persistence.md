# Harness owns training, data loading, and artifact persistence

ML Autoresearch prioritizes safe, auditable research iteration over arbitrary candidate authority. Candidate Experiments may express research variation only through an allowlisted Candidate Experiment Contract; the trusted Harness owns training loops, data loading, validation, execution policy, local run metadata, and artifact persistence.

If new research freedom is needed, it is added as an explicit Harness-owned parameter or audited capability rather than by allowing arbitrary candidate filesystem, network, dataset, training-loop, persistence, or external service access. Earlier sketches mentioned MLflow; the current implementation uses Harness-owned local metadata and artifacts.
