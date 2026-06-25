# Filesystem Research Problem packages provide checked Specs

ML Autoresearch accesses Research Problem-specific trusted code through filesystem Research Problem packages that expose a checked Research Problem Spec provider. Research Problem packages are private research code, not PyPI packages or public marketplace plugins.

A Harness-owned configuration file names the active Research Problem package using:

1. the Research Problem id, such as `ground_camera_contrail_detection`;
2. the package root path on the local filesystem;
3. a provider target inside that package, such as `gvccs.research_problem:build_spec`;
4. the expected Research Problem contract version;
5. Research Problem data configuration, such as the dataset root.

The Harness resolves the configured package root, imports the provider, calls it to obtain a `ResearchProblemSpec`, validates that the Spec matches the configured id and a supported contract version, validates declared brief/profile paths, and registers the Spec in the Research Problem Spec Registry. Operation-specific training/evaluation code enforces adapter support instead of treating every possible adapter as mandatory at registration.

After registration, reusable Harness code uses Research Problem-specific behavior only through the checked Spec interface and trusted adapters, not direct imports of problem-specific modules such as GVCCS dataset code.

The configured filesystem path identifies source location, not semantic interface. Candidate Experiments remain untrusted and cannot provide Research Problem packages, mutate the registry, add adapters, or bypass the Candidate Experiment Contract.

Run metadata records Research Problem provenance for reproducibility and auditability, including the Research Problem id, Spec version, contract version, provider target, resolved source path, and, when available, source-control information such as git commit and dirty state. The Candidate Execution Boundary receives the same trusted Research Problem package by mounting or copying the configured package root into the container and configuring the Python import path for the Harness-owned operation.

The deletion test for this decision is that removing the Ground-Camera Contrail Detection package leaves the reusable Harness package importable and testable, while Runs for that Research Problem fail only at provider loading/registration.
