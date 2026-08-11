# ADR 0011: Manage long Run execution by stable Run identity and idempotent terminalization

## Status

Accepted.

## Context

A synchronous caller can disappear while a Docker Candidate Execution continues. The container may finish and write complete artifacts, but caller-owned finalization then leaves Run metadata in `training` and omits the terminal Research Ledger event. Re-running the Candidate risks duplicate training, while independently writing metadata and ledger events permits duplicate or conflicting terminal records.

## Decision

The Harness validates and copies a Candidate into a stable Run before smoke starts. A detached Harness supervisor recorded in `execution.json` owns both smoke and training; foreground callers follow that supervisor rather than owning either operation. Docker training containers remain inspectable until terminalization instead of using `--rm`, and their identities and attempts are recorded durably.

`execution.json` exposes the pre-training `smoke_testing` phase. If a caller disappears during smoke, observation and reconciliation continue against the same Run ID rather than resubmitting the Candidate. A vanished smoke supervisor with no active container is terminalized as a Harness failure; reconciliation never starts another smoke test or training operation.

`run-status` observes an existing Run without launching work. `reconcile-run` operates only on an existing Run, validates required artifacts and numerical state, and never retrains. Open autonomy actions map a previously submitted Candidate to its existing Run ID and reconcile that Run rather than submitting another one.

Terminal metadata and Research Ledger writes are serialized by per-Run and terminal-ledger locks. Terminal event append is compare-before-append and idempotent. Reconciliation repairs either side of a partial metadata/event transition, rejects conflicting or duplicate terminal events, and removes recorded Docker containers only after terminalization.

## Consequences

- Caller interruption during smoke or training does not stop a managed Run or require duplicate submission.
- Operators can distinguish supervisor, container, artifact, and terminal Run state.
- Exited containers may remain briefly until finalization or reconciliation removes them.
- A missing execution/container record with incomplete artifacts fails as a Harness failure; reconciliation does not guess by retraining.
- Resource Failure retries remain attempts within one stable Run. Candidate non-finite failures are not Resource Failures and are not retried.
