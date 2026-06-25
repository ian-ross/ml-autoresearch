# Per-pixel Auxiliary Targets in the Candidate Experiment Contract

ML Autoresearch supports a top-level `auxiliary_targets` list for provider-declared per-pixel Auxiliary Targets. The Harness derives expected auxiliary outputs from the Resolved Manifest and passes them through `output_spec["auxiliary_outputs"]`; requested auxiliary outputs are required, output names must match exactly, and the primary output remains mandatory.

The generic contract is provider-driven: target names, output names, loss names, and shapes must be advertised by the active Research Problem Spec. Candidate Experiments may return requested auxiliary logits, but must not derive auxiliary target tensors or implement auxiliary losses. Auxiliary and total losses are recorded separately; primary model comparison remains based on the Research Problem primary output and selection metric. Post-Run Evaluation tolerates auxiliary outputs while evaluating the primary prediction.

GVCCS-specific Line and Boundary Target semantics are Research Problem provider details and are documented temporarily in `docs/gvccs-features.md`.
