# Autonomous Research Campaign Plan

This document records the design discussion for moving ML Autoresearch from Human-Guided Research Iterations toward bounded Autonomous Research Iterations and eventually multi-week Research Campaigns.

## Goal

ML Autoresearch should let an agent explore model space for a Research Problem by proposing, implementing, running, evaluating, documenting, and iterating on Candidate Experiments within a trusted Harness. Ground-Camera Contrail Detection using the GVCCS Dataset remains the proving-ground Research Problem; Satellite Contrail Detection is a later Research Problem intended to reuse the same infrastructure once the loop is mature.

## Autonomy scope

The target is **Run-loop autonomy**. During autonomous operation the agent may:

- inspect prior Results, Research Notes, Campaign Reports, and Research Ledger events;
- write an Experiment Proposal before generating candidate code;
- create Candidate Experiments and Architecture Helpers within the Candidate Experiment Contract;
- submit Runs and observe Results;
- invoke approved, bounded Post-Run Evaluations after writing Evaluation Requests;
- classify failed Runs;
- write Research Notes with provenance-tracked Research Figures;
- append Research Ledger events through Harness-owned commands;
- create Capability Requests when useful research is blocked;
- write Campaign Reports and pause when policy requires it.

The agent must not:

- change the Research Problem;
- add datasets or change locked evaluation policy;
- expand the Candidate Experiment Contract;
- approve pretrained weights;
- modify Harness code, tests, or trusted infrastructure docs;
- use candidate code to work around Harness limitations.

Harness changes happen only in a separate human-supervised process in response to Capability Requests.

## Candidate authority and covert workarounds

Candidate Experiments remain constrained by the Candidate Experiment Contract. If a desired experiment requires capability outside the contract, the agent must create a Capability Request rather than attempting to emulate, tunnel, hide, or approximate the missing capability in candidate code.

Forbidden covert workarounds include:

- candidate-owned data loading or dataset path probing;
- custom training loops, losses, samplers, or transforms in helper files;
- runtime downloads or arbitrary checkpoint references;
- writing side-channel artifacts for unofficial evaluation;
- using model code to inspect the filesystem or environment;
- encoding multiple experiments into one Candidate Experiment to bypass Experiment Batch limits;
- disguising architecture-independent policy changes as Model Architecture code;
- helper modules that do anything other than define architecture layers, blocks, or model-composition code.

## Experiment structure

Every Autonomous Research Iteration requires a pre-code **Experiment Proposal** stored with the Candidate Experiment source, typically as `PROPOSAL.md`, and indexed by the Research Ledger. The proposal records:

- hypothesis;
- Comparison Target;
- expected effect;
- implementation sketch;
- contract features used;
- budget requested;
- success criteria;
- fallback or next decision if it fails.

`README.md` remains separate from `PROPOSAL.md`: the Candidate README summarizes the implemented Model Architecture, manifest choices, contract assumptions, and known limitations. Research Notes are written after Results and record outcome, figures, and decisions.

An Autonomous Research Iteration normally proposes one Candidate Experiment, but may propose a bounded Experiment Batch when parallel variants are tied to one hypothesis. Maximum parallel submissions are governed by Harness policy.

## Research progress

Research Progress is a portfolio objective, not leaderboard-only optimization. The agent should improve the best validated Result when possible while also reducing uncertainty about which model families, input modes, losses, data policies, and training policies merit further investment.

Each proposal declares a Comparison Target that may be the global best Run, a family baseline, or a direct parent Run. Global-best comparison should still be reported, but it is not always the primary control.

Stalled Research Progress is assessed using both metric plateau and structured self-assessment about recent hypothesis value, repetition, blocked next steps, and whether new Capability Requests are needed.

## Research memory and reporting

The Research Ledger is the canonical append-only record of campaign activity. When Harness-owned automation or an agent records campaign events, it must use `ml-autoresearch record-research-event` or the `record_research_event` API rather than editing `research-ledger.jsonl` directly.

The canonical ledger file is `research-ledger.jsonl` at the repository or campaign workspace root. Each line is one validated JSON event with a Harness-generated `created_at` timestamp.

Initial event types:

- `proposal_created`
- `candidate_created`
- `candidate_submitted`
- `run_started`
- `run_completed`
- `run_failed`
- `research_note_written`
- `capability_request_created`
- `evaluation_requested`
- `evaluation_completed`
- `campaign_report_written`
- `agent_handoff_ingested`
- `campaign_paused`

Example:

```bash
ml-autoresearch record-research-event \
  --event-type run_started \
  --field run_id=run_123 \
  --field candidate_id=candidate_abc
```

`research_note_written` events include `note_path` and may include `run_id` for a single-Run note. Notes with Research Figures may also include `figure_provenance_path` pointing to the note section or sidecar provenance file, or `figure_provenance` metadata containing the same validated source Run/Evaluation IDs, source artifact paths, and selection reasons documented in `research-notes/README.md`.

Campaign Reports and Campaign Pause Conditions are documented in `docs/campaign-report-format.md`. Prefer `ml-autoresearch record-campaign-report` and `ml-autoresearch pause-campaign` over the generic event command for those event types so the approved pause vocabulary and optional report linkage are used consistently.

`agent_handoff_ingested` events audit Agent Handoff Ingestion separately from Candidate Experiment execution or Post-Run Evaluation execution. They require `handoff_type`, `artifact_id`, `source_path`, and `canonical_path`; supported handoff types are `candidate_submission`, `evaluation_request`, `capability_request`, `research_note`, and `campaign_report`. They may include type-specific linkage fields such as `candidate_id`, `request_id`, `note_path`, `report_path`, and `run_id`.

## Run lifecycle emission

The Run lifecycle inside `ml_autoresearch.runs` automatically emits Research Ledger events through the same validated `record_research_event` API as the CLI:

- `submit_candidate` records a `candidate_created` event as soon as the
  candidate source passes validation. Rejected Runs do not produce either
  `candidate_created` or `candidate_submitted` events. For Repair Candidates,
  this event includes `repair_lineage` with the original proposal, original
  candidate, motivating failed Run, failure classification, and the explicit
  preservation flags for the original hypothesis and Comparison Target.
- `submit_candidate` records a `candidate_submitted` event when a candidate
  also passes smoke testing. Smoke-failed Runs do not produce a
  `candidate_submitted` event.
- `run_candidate_with_synthetic_fixture` and `run_candidate_with_gvccs_data`
  record `run_started` once training begins, then either `run_completed`
  (with the final-metrics artifact path) or `run_failed` (with the failure
  reason and approved `failure_classification`) when the Run terminates.
- `run_post_run_evaluation` records `evaluation_requested` and
  `evaluation_completed` after a valid Evaluation Request creates bounded
  request-linked artifacts.
- `evaluate-run` creates an implicit manual Evaluation Request artifact at
  `outputs/evaluations/<evaluation_id>/evaluation_request.json` and records the
  same `evaluation_requested` / `evaluation_completed` ledger pair for the
  durable Post-Run Evaluation artifacts it writes.

By default, lifecycle events are appended to `<runs_root>/../research-ledger.jsonl`, which places the ledger at the campaign workspace root next to the local `runs/` artifact tree. Pass `--ledger-path` to `submit-candidate`, `run-candidate`, and `evaluate-run`, or the corresponding `ledger_path` keyword argument to the API, to override the destination (for example in tests).

Later event types may include:

- `research_figure_created`
- `resource_retry`
- `comparison_recorded`
- `budget_updated`

Research Notes should include human-readable figures. Figures may be existing Run/Post-Run Evaluation artifacts or generated report figures, but each Research Figure must record provenance: source Run ID or Evaluation ID, source artifact path, and reason for selection.

Campaign Reports should be produced on a fixed cadence and in response to events such as pause conditions, significant improvements, repeated failures, or high-priority Capability Requests.

## Capability Requests

A Capability Request is a structured, human-gated request to expand Harness-owned contract surface, approved resources, or operational policy when current capabilities block useful Research Progress. Requests should be written using `docs/capability-request-format.md`.

Capability Requests are not self-approving. The agent may continue using existing Candidate Experiment Contract choices while a request is pending. Candidate authority requested should normally be `none`; prefer Harness-owned parameters or approved resources.

## Capability Slice strategy

The Candidate Experiment Contract should expand through small Capability Slices that unlock meaningful experiment families while preserving Harness ownership.

Near-term priority after the Autonomy Smoke Loop:

1. Augmentation Policy presets.
2. Additional losses plus scheduler or early stopping support.
3. Boundary Target.
4. Temporal Input.
5. Approved Pretrained Encoders.

Augmentation Policy is Research Problem-owned because valid transforms depend on data modality and dataset semantics. Future implementation should separate reusable infrastructure from trusted Research Problem Repositories, as recorded in ADR 0006.

## Campaigns, budgets, and pauses

A Research Campaign is a bounded autonomous research effort made of many Autonomous Research Iterations. It is governed by hierarchical budgets:

- campaign-level budget;
- batch/concurrency budget;
- per-Run wall-clock and resource budget;
- repair-attempt budget;
- storage/artifact budget;
- reporting cadence.

Resource Budget Policy is Harness-owned and includes GPU assignment, CPU memory, shared memory, process limits, concurrency, model/training bounds, and bounded retry behavior after Resource Failures.

Docker can constrain CPU RAM, shared memory, process limits, CPUs, and GPU device assignment. It generally cannot impose a simple reliable CUDA VRAM quota per container. Practical GPU-memory controls include one Run per GPU or explicit GPU slot scheduling, Harness-enforced bounds on batch size/image size/clip length/parameter count, mixed precision, smoke-test memory checks where feasible, bounded OOM retry, and MIG partitioning where available.

Campaign Pause Conditions use the approved values documented in `docs/campaign-report-format.md`:

- `budget_exhausted`;
- `repeated_failures`;
- `repeated_resource_failures`;
- `stalled_research_progress`;
- `too_many_pending_capability_requests`;
- `storage_risk`;
- `scheduled_check_in`.

## Failure handling and repairs

After a failed Run, the autonomous loop records a Run Failure Classification before acting. Failure classes use the approved Run Failure Classification vocabulary from `docs/run-lifecycle.md`: `candidate_bug`, `contract_violation`, `resource_failure`, `harness_failure`, `bad_research_result`, and `unknown`.

A submitted Candidate Experiment is never overwritten. Bug fixes are submitted as distinct Repair Candidates with repair lineage. Initial autonomous policy allows at most two Repair Candidates per original proposal. A Repair Candidate must preserve the original hypothesis and Comparison Target; changing the scientific idea requires a new Experiment Proposal and Candidate Experiment lineage.

GPU out-of-memory and similar allocation failures are Resource Failures. The Harness may perform bounded retry with a smaller effective batch size before failing the Run. When batch size is lowered, `resolved_manifest.yaml` records both `training.batch_size_requested` and `training.batch_size_effective`, and `run_metadata.json` records the Resource Failure retry lifecycle.

## Post-Run Evaluations

The agent may invoke approved, bounded Harness-owned Post-Run Evaluations on prior Runs. Each autonomous evaluation requires an Evaluation Request recording:

- target Run ID;
- approved evaluation mode;
- diagnostic question;
- expected decision impact;
- bounded diagnostic parameters;
- artifact/resource budget.

Evaluation Requests may choose bounded Harness-approved diagnostic parameters such as thresholds, threshold sweep bounds, evaluation batch size, artifact counts, and failure buckets. Whole-Validation Failure Analysis is the umbrella Post-Run Evaluation diagnostic for a completed Run across the Working Validation Split. The manual/operator `evaluate-run` surface runs that diagnostic directly, writes metric artifacts under `runs/<run_id>/outputs/evaluations/<evaluation_id>/`, and stores its implicit manual request as `evaluation_request.json` beside those artifacts. The autonomous request-gated `run-post-run-evaluation` surface validates explicit `threshold_sweep` and `failure_bucket_review` requests, which select bounded parts of the same Whole-Validation Failure Analysis capability and write request-linked artifacts under `runs/<run_id>/evaluations/<evaluation_id>/`.

## Validation overfitting control

Autonomous exploration uses an agent-visible Working Validation Split. Locked Evaluation Split access is reserved for human-gated milestone checks to reduce validation overfitting risk. Locked-evaluation Results may be summarized to the agent in Campaign Reports after human-triggered milestone checks, but access frequency remains limited.

## Autoresearch skills

The Autoresearch Skill Set should be hierarchical and use progressive disclosure. A campaign-manager skill should orchestrate focused skills for:

- reading current context/ledger/status;
- proposing experiments;
- implementing candidates;
- submitting and observing Runs;
- classifying failures;
- writing Evaluation Requests;
- writing Research Notes with figures;
- appending ledger events;
- creating Capability Requests;
- writing Campaign Reports;
- pausing when required.

## Autonomy Smoke Loop

Before broader contract expansion, run a short Autonomy Smoke Loop using the current narrow Candidate Experiment Contract. The smoke loop should prove disciplined behavior rather than metric improvement.

Success criteria:

- pre-code Experiment Proposals are created;
- candidates stay within the Candidate Experiment Contract;
- valid candidates or bounded Repair Candidates are submitted;
- failed Runs are classified correctly;
- Research Notes include figure provenance;
- Research Ledger events are appended through a Harness-owned command;
- Capability Requests are created when blocked;
- Campaign Reports are produced;
- Campaign Pause Conditions are respected.

## First implementation milestone

The first implementation milestone is the Research Ledger vertical slice:

1. Define `research-ledger.jsonl` and initial event schemas.
2. Add a Harness-owned `record-research-event` CLI/API.
3. Validate and append events atomically.
4. Test schema validation and append behavior.
5. Then add candidate-local `PROPOSAL.md` handling and initial autoresearch skills.
