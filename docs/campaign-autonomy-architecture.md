# Campaign autonomy architecture

This document describes the current ML Autoresearch campaign/autonomy architecture.

## Research Ledger

The canonical Research Ledger is `research-ledger.jsonl`, an append-only JSONL file written through Harness-owned APIs such as `record-research-event`. Operators may override ledger paths on supported commands, including `submit-candidate`, `run-candidate`, `evaluate-run`, and `run-post-run-evaluation`. In workspace configuration, `candidate_execution.ledger_path` is configurable but must resolve inside the Research Workspace Root; large Run artifacts may use external `runs_root`, but campaign state stays workspace-local.

Implemented event families include proposal/candidate/run lifecycle, post-run evaluation, capability request, agent handoff ingestion, campaign report, campaign pause/resume, and Experiment Batch events such as `experiment_batch_created`, `experiment_batch_completed`, `batch_candidate_created`, and `batch_run_started`.

## Campaign loop

A campaign iteration observes prior Runs, evaluations, notes, and ledger state; proposes one next research action; writes one primary handoff artifact inside the Agent Workspace; then stops. The Harness ingests the handoff outside the Agent Control Boundary and records ledger/index updates.

The one-primary-handoff rule prevents multiple competing next actions in one autonomy step. Supported primary handoff types include candidate submissions, research notes, capability requests, evaluation requests, campaign reports, and experiment batch submissions.

## Agent Control Boundary

The inner agent receives curated read-only context: reference docs, run history, batch history, research notes, Research Problem briefs, dataset profile artifacts, and optional explicit `/data` mounts. It can write drafts and handoffs under its Agent Workspace, but must not execute Candidate Experiments, mutate the ledger, run Docker/GPU tools, or treat local analysis as authoritative Results.

Authoritative execution happens through Harness-owned commands outside the VM. Handoff ingestion is not Run execution and is not Post-Run Evaluation execution.

## Autonomy commands

- `prepare-agent-boundary` creates curated snapshots, installs packaged Autoresearch skills, and writes the generated pi-fort configuration.
- `autonomy-step` prepares a prompt, runs or supports a bounded agent step, ingests exactly one handoff when present, and can execute one next Harness-owned action with `--execute-next-action`.
- `run-autonomous-iteration` wraps bounded repeated operation around the same handoff and execution rules.

Each step records recovery state: prompt, result JSON, handoff markers such as `.INGESTED.json`, and ledger events.

## Reports, pauses, and resumes

Campaign reports summarize evidence, recommended next action, budget state, and pause conditions. `pause-campaign` records `campaign_paused` with an approved reason. `resume-campaign` records `campaign_resumed`; newer resume events clear prior scheduled-check-in or resolved-review blockers in later autonomy prompts.

See `docs/campaign-report-format.md` for the report format.

## Requests and evaluations

Capability Requests and Evaluation Requests are not self-approving. They create auditable handoffs for Harness or human work.

- Capability Request format: `docs/capability-request-format.md`
- Evaluation Request format: `docs/evaluation-request-format.md`
- Post-Run Evaluation artifacts are run-scoped under `outputs/evaluations/<evaluation_id>/`.

## Failure classification and repair lineage

Run failures use the vocabulary in `docs/run-lifecycle.md`. Repair Candidates are new Candidate Experiments with explicit lineage to the original proposal/candidate/run. A repair may fix a candidate bug or contract issue while preserving the original hypothesis and comparison target; scientific changes require a new proposal.

## Current non-goals

- Unbounded autonomous continuation after a pause decision.
- Candidate-owned training/evaluation, data loading, ledger writes, or runtime weight downloads.
- Treating GVCCS-specific behavior as generic Harness behavior; Research Problem-specific features belong behind provider specs and docs such as `docs/gvccs-features.md`.
