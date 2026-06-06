# Filesystem Research Problem packages provide checked Specs

ML Autoresearch will access Research Problem-specific trusted code through filesystem Research Problem packages that expose a checked Research Problem Spec provider. Research Problem packages are private research code and are not expected to be distributed through PyPI or a public plugin marketplace.

A Harness-owned configuration file will name the active Research Problem package using at least:

1. the Research Problem id, such as `ground_camera_contrail_detection`;
2. the package root path on the local filesystem;
3. a provider target inside that package, such as `gvccs.research_problem:build_spec`;
4. the expected Research Problem contract version;
5. Research Problem data configuration, such as the dataset root.

The Harness will resolve the configured package root, import the configured provider, call it to obtain a `ResearchProblemSpec`, validate that the returned Spec matches the configured id and a Harness-supported contract version, validate that required adapters are present, and register the Spec in the Research Problem Spec Registry. After registration, reusable Harness code should interact with Research Problem-specific behavior only through the checked Spec interface and its trusted adapters, not through direct imports of problem-specific modules such as GVCCS dataset code.

The configured filesystem path is therefore the source-location mechanism, not the semantic interface. The semantic interface is the Research Problem contract exposed by the returned Spec. Candidate Experiments remain untrusted and cannot provide Research Problem packages, mutate the registry, add adapters, or bypass the Candidate Experiment Contract.

Run metadata will record Research Problem provenance for reproducibility and auditability, including the Research Problem id, Spec version, contract version, provider target, resolved source path, and, when available, source-control information such as git commit and dirty state. The Candidate Execution Boundary must receive the same trusted Research Problem package, for example by mounting or copying the configured package root into the container and configuring the Python import path for the Harness-owned training or evaluation operation.

The deletion test for this decision is that removing the Ground-Camera Contrail Detection package should leave the reusable Harness package importable and testable, while Runs for that Research Problem fail only at the point where the configured Research Problem provider cannot be loaded or registered.
