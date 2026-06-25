# Research Problem Repositories are Research Workspace Roots

ML Autoresearch separates reusable Harness code from live Research Loop state by making each trusted Research Problem Repository both the Python package boundary for problem-specific provider code and that problem's Research Workspace Root.

Workspace-bound Harness commands use `--workspace-root` (defaulting to the current directory), read one canonical `ml-autoresearch.toml`, and keep fixed research-state paths such as `EXPERIMENT_INDEX.md`, `research-ledger.jsonl`, `candidates/`, `research-notes/`, and agent handoff directories under that root. `runs_root` may point to external storage for large Run artifacts. `ledger_path` is configurable but must resolve inside the Research Workspace Root because the ledger is canonical campaign state.

The Agent Control Boundary must not mount the full Research Problem Repository. Instead, `prepare-agent-boundary` generates a curated Agent Research Problem Snapshot containing only declared Research Problem Brief documents, Dataset Profile Artifacts, and their index, while the Candidate Execution Boundary receives the trusted package root needed for Harness-side imports.

Harness-owned agent resources and container build recipes are packaged with `ml-autoresearch`; built Agent Runtime Image assets and runtime validation state live under hidden workspace operational state such as `.ml-autoresearch/`. Docker runner images use workspace- and Harness-version-specific tags, and runtime operations require a validation stamp showing that configured images match the Harness version or explicit development source override.
