# Experiment Batches

An Experiment Batch is a synchronous, Harness-owned way to run a small set of related Candidate Experiments under one shared hypothesis and one shared comparison target.

## Constraints

- Batch size is capped at 4 Candidate Experiments.
- Parallel execution is capped at 4 Runs.
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

`BATCH_PROPOSAL.md` records the shared hypothesis, shared comparison target, per-candidate rationale, decision criteria, success criteria, and requested budget/concurrency subject to Harness caps.

## Artifacts

Batch-level artifacts live under `batches/<batch_id>/` and include `batch_metadata.json` plus a copy of `BATCH_PROPOSAL.md`. Per-Run artifacts remain under `runs/<run_id>/`; each Run metadata records `batch_id` and `batch_candidate_id`.

## Observation

Use `list-batches` to list batch artifacts and `batch-summary` to inspect per-candidate Run status, failure classification, and metrics where available.
