# Experiment Batches

An Experiment Batch is a synchronous, Harness-owned way to run a small set of related Candidate Experiments under one shared hypothesis and comparison target.

## Constraints

- Batch size is capped at 4 Candidate Experiments.
- Parallel execution has a hard cap of 4 Runs and a conservative workspace default of 1; `[candidate_execution].max_parallel_runs` may lower or raise the active cap within that range only after operator resource review.
- GPU placement is Harness-owned. `[candidate_execution].docker_gpu_device` can pin Docker Runs to one reviewed device; Candidate code and Agent handoffs cannot select devices.
- A batch is one primary research action, not an arbitrary sweep.
- Static Candidate Experiment validation is all-or-nothing before execution starts.
- Once execution starts, a failed Run does not cancel sibling Runs.
- The Harness blocks until all batch Runs complete or fail.

## Handoff shape

```text
experiment_batch/
├── BATCH_PROPOSAL.md
└── candidates/
    ├── variant_a/
    │   ├── manifest.yaml
    │   ├── model.py
    │   └── README.md
    └── variant_b/
        ├── manifest.yaml
        ├── model.py
        └── README.md
```

`BATCH_PROPOSAL.md` records the shared hypothesis, comparison target, per-candidate rationale, decision criteria, success criteria, and requested budget/concurrency subject to Harness caps. The Harness validates these declarations before any Run is created.

Agents can package a batch as one primary handoff with `prepare-experiment-batch-submission`; the Harness ingests it with `ingest-experiment-batch-submission` or as part of an Autonomy Step. Ingested source batches are copied to `experiment-batches/<batch_submission_id>/`; execution artifacts are still created under `batches/<batch_id>/`.

## Execution

`run-experiment-batch` executes each candidate through the configured Research Problem in the workspace-local `ml-autoresearch.toml`. Ingested Agent handoffs use the Harness-owned `candidate_execution.max_parallel_runs` value; direct operator commands may request a lower cap but remain hard-capped at 4. Batch execution is synchronous from the caller's perspective: the command returns only after all sibling Runs complete or fail. The Harness records `experiment_batch_created`, `batch_candidate_created`, `batch_run_started`, and `experiment_batch_completed` Research Ledger events while preserving existing per-Run audit events.

Every successful or failed training attempt writes `outputs/resource_profile.json` and a batch-size-keyed record under `outputs/resource_profiles/`. Profiles include device identity, peak PyTorch CUDA allocated/reserved memory when available, phase timing, throughput, and failure evidence. Resource-retry metadata remains Harness-owned in Run metadata. These measurements support operator concurrency decisions; they do not grant Candidate code resource-policy authority.

## Artifacts

Batch-level artifacts live under `batches/<batch_id>/` and include `batch_metadata.json` plus a copy of `BATCH_PROPOSAL.md`. Per-Run artifacts remain under `runs/<run_id>/`; each Run metadata records `batch_id` and `batch_candidate_id`.

## Observation

Use `list-batches` to list batch artifacts and `batch-summary` to inspect per-candidate Run status, failure classification, and available metrics.
