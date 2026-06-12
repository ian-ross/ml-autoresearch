# Agent Control Boundary

The Agent Control Boundary is the permission boundary for an autonomous agent
running inside the VM. It protects infrastructure authority: the agent cannot
control host shell, Docker, GPUs, cluster schedulers, secrets, cloud credentials,
or Harness-owned Run execution. It is not primarily a dataset hiding mechanism
for overfitting control; validation authority and authoritative Results remain
owned by the Harness.

## Inner-agent prompt contract

Prompt the inner agent with these operational rules:

- The Agent Workspace is the current writable directory inside the VM.
- Use `ml-autoresearch-agent`, not `ml-autoresearch`, inside the VM.
- Draft Candidate Experiments and handoff artifacts only in the Agent Workspace.
- Review read-only reference, history, docs, and approved data mounts before
  proposing work.
- Do not modify Harness code, canonical Research Ledger files, canonical
  Experiment Index files, or mounted history/reference/data.
- Do not seek hidden authority through filesystem probing, candidate helper
  files, side-channel artifacts, environment inspection, subprocesses, or
  network workarounds.

## Autonomous iteration loop

`ml-autoresearch run-autonomous-iteration` is the bounded loop command built on
Autonomy Steps. It requires `--notify-email` and at least one of `--max-steps`
or `--max-duration`; if both limits are provided, the first reached limit stops
the loop. The duration syntax is `N`, `Ns`, `Nm`, or `Nh`. Time limits are checked
between steps: an in-flight agent step, Candidate Experiment Run, or Post-Run
Evaluation is allowed to finish before the loop decides whether to start another
step.

The loop always executes Harness-owned next actions for executable handoffs. It
continues after Research Notes, completed Candidate Runs, and completed
Post-Run Evaluations. It stops for step/time limits, agent failure, ingestion
failure, execution failure, no handoff, Capability Requests, Campaign Reports,
campaign pauses, and other human-review outcomes. Before starting, it rejects a
dirty Agent Workspace containing un-ingested primary handoff artifacts so stale
manual outputs cannot be mistaken for new autonomous work.

On completion the loop writes `agent-work/autonomous-iteration-result.json` and
sends a Mailjet plain-text completion email. Mailjet settings live in local
`notification.toml` at the project root:

```toml
[mailjet]
api_key = "..."
api_secret = "..."
from_email = "verified-sender@example.com"
from_name = "ML Autoresearch"
```

Real credentials should be kept local and out of version control.

## Autonomy Step operator workflow

`ml-autoresearch autonomy-step` is the operator command for one Harness-owned
Autonomy Step. As described by [ADR 0007](adr/0007-autonomy-steps-ingest-one-agent-handoff-before-execution.md), an
Autonomy Step refreshes the Agent Control Boundary, invokes the agent once with
a generated `prompt.txt`, performs Agent Handoff Ingestion for exactly one
primary handoff outcome, writes `agent-work/autonomy-step-result.json`, and then
stops unless the operator explicitly enables one bounded Harness-owned next
action.

The generated `prompt.txt` tells the inner agent to read `AGENTS.md`, use the
campaign-manager skill, choose exactly one primary research handoff outcome, and
stop after the first primary outcome. The one-primary-handoff rule applies to
Candidate Submissions, Research Notes, Capability Requests, Evaluation Requests,
and Campaign Reports. If the agent is blocked, it writes either one Campaign
Report or one Capability Request, not both. If no useful handoff is safe, it may
stop without producing a handoff.

Agent Handoff Ingestion is not Candidate Experiment Run execution and is not
Post-Run Evaluation execution. Ingestion is the Harness-owned audit step that
validates one artifact, copies it from the Agent Workspace to the canonical
project location, rejects duplicate canonical destinations, records Research
Ledger events, updates canonical indexes when applicable, and writes an
`.INGESTED.json` marker beside the source artifact only after canonical state is
updated. Candidate Experiment execution happens later through the Candidate
Experiment Runner. Post-Run Evaluation execution happens later through the
Post-Run Evaluation subsystem. `autonomy-step --execute-next-action` may execute
one selected Harness-owned next action after successful ingestion: submit and
run one Candidate Experiment Run for an ingested Candidate Submission, or run
one Post-Run Evaluation for an ingested Evaluation Request. If the operator ran the
Autonomy Step without `--execute-next-action`, `ml-autoresearch
execute-next-action` reads the previous `agent-work/autonomy-step-result.json`
and executes the same outstanding Harness-owned next action later. Candidate
Run next actions use the project-root `candidate-execution.toml` Candidate
Execution Boundary policy rather than Agent Control Boundary settings.

The result file, `agent-work/autonomy-step-result.json`, records the agent
command, return code, ingestion status, handoff type, canonical path, next
action, optional next-action execution result, and failure reason. The default
Pi invocation stores Pi session JSONL files in `agent-sessions/`, a gitignored
sibling of `agent-work/`. Inspect the result file, the source-side ingestion
marker, and the matching Pi session record before starting another Autonomy Step
or graduating the workflow into a future bounded autonomous iteration loop.

### Manual test checklist

Use a disposable workspace and run one case per Autonomy Step:

- Candidate Submission: agent creates one queued Candidate Experiment under
  `submissions/`; ingestion copies it to `candidates/`, records the handoff, and
  reports `next_action: run_candidate` without submitting or training a Run unless
  `--execute-next-action` is set. If a prior execution only reached an accepted
  Run, `execute-next-action` must continue that accepted Run rather than treating
  acceptance as a completed `run_candidate` action.
- Research Note: agent creates one Research Note under `research-notes/`;
  ingestion copies it to canonical `research-notes/`, updates
  `EXPERIMENT_INDEX.md`, records the handoff, and writes an ingestion marker.
- Capability Request: agent creates one Capability Request under
  `capability-requests/`; ingestion copies it to canonical state and stops for
  human-gated review rather than self-approving contract or policy expansion.
- Evaluation Request: agent creates one Evaluation Request under
  `evaluation-requests/`; ingestion records `next_action:
  run_post_run_evaluation` without running the Post-Run Evaluation unless
  `--execute-next-action` is set.
- Campaign Report: agent creates one Campaign Report under `campaign-reports/`;
  ingestion copies it and stops for human campaign review or pause handling.
- No handoff: agent exits successfully without creating primary handoff
  artifacts; `autonomy-step-result.json` reports `no_handoff` and
  `stop_for_human`.
- Duplicate artifact: the canonical destination already exists; ingestion fails before copying, ledger updates, index updates, or marker creation.
- Multi-handoff failure: the Agent Workspace contains more than one primary
  handoff outcome, or more than one artifact for a primary handoff type;
  ingestion fails and the operator must resolve the workspace before continuing.

## Filesystem layout

### Writable Agent Workspace

The Agent Workspace is the current writable directory. It is prepared by the
Harness before the agent starts and contains these writable subdirectories:

- `drafts/candidates/` — mutable draft Candidate Experiment directories.
- `submissions/` — the Candidate Submission Queue for finalized handoffs.
- `research-notes/` — draft Research Notes and note assets.
- `capability-requests/` — requests for Harness-owned contract or policy
  expansion.
- `evaluation-requests/` — requests for Harness-owned Post-Run Evaluations.
- `campaign-reports/` — campaign status and pause reports.
- `scratch/` — temporary analysis files, plots, summaries, and toy checks.

`submissions/` entries are immutable once created. The agent must create a new
submission for a repair or revision rather than editing an existing queued
entry. Submissions are ingested by the Harness outside the Agent Control Boundary,
where validation, canonical ledger/index updates, and Run scheduling occur.

The Harness also writes `agent-work/AGENTS.md` with the Agent Control Boundary
path map. It tells the inner agent how to translate project-root paths mentioned
in Autoresearch skills into boundary mounts, for example `CONTEXT.md` to
`/reference/CONTEXT.md`, `docs/` to `/docs/`, and prior `research-notes/` to
`/history/research-notes/`, while keeping new draft notes under the writable
workspace `research-notes/` directory.

`prepare-agent-boundary`, `autonomy-step`, and `run-autonomous-iteration`
require an explicit `[research_problem]` provider in root
`candidate-execution.toml`. Agent handoff/autonomy flows do not fall back to a
built-in/default Research Problem, because the active provider is the source of
agent-visible Research Problem Brief metadata. During setup the Harness loads
that provider, validates its declared brief documents, mounts the provider
package read-only at `/research-problem`, and writes a progressive-disclosure
brief index into both `agent-work/AGENTS.md` and
`agent-work/RESEARCH_PROBLEM_BRIEF_INDEX.md`. Each index entry includes the
document name/role, optional summary, required marker, mounted path, and a
simple read command such as `cat /research-problem/brief/overview.md`. The full
brief documents are not embedded by default; the agent starts from the index and
selectively reads only the deeper documents needed for the current Candidate
Experiment.

The Harness installs the reviewed Autoresearch Skill Set from
`docs/autoresearch-skills/` into `agent-work/.pi/skills/` during setup so the
inner agent can use the campaign-manager and focused autoresearch skills as
active Pi skills inside the VM.

### Read-only mounts

The VM exposes these read-only paths:

- `/reference` — the Agent Reference Snapshot generated at setup time.
  - `/reference/CONTEXT.md` — canonical domain language snapshot.
  - `/reference/EXPERIMENT_INDEX.md` — canonical experiment index snapshot.
- `/history` — prior research material exposed for review.
  - `/history/research-ledger.jsonl` — Research Ledger snapshot.
  - `/history/candidates/` — prior Candidate Experiment sources, when
    available.
  - `/history/runs/` — prior Run summaries/artifacts exposed by the Harness.
  - `/history/batches/` — prior Experiment Batch summaries/artifacts exposed by the Harness.
  - `/history/research-notes/` — prior Research Notes.
- `/docs` — trusted project documentation, including the Candidate Experiment
  Contract, Run lifecycle, request/report formats, and agent skill docs.
- `/research-problem` — the active configured Research Problem provider package,
  including any declared Research Problem Brief documents.
- `/data` — approved read-only Research Problem data mounts, when policy
  permits.

## Approved commands

Inside the VM, allowed ML Autoresearch operations go through the agent-safe CLI
wrapper:

- `ml-autoresearch-agent validate-candidate`
- `ml-autoresearch-agent prepare-candidate-submission`
- `ml-autoresearch-agent list-runs`
- `ml-autoresearch-agent run-summary` / `ml-autoresearch-agent get-run-summary`
- `ml-autoresearch-agent get-best-runs`

Capability Requests, Evaluation Requests, Research Notes, and Campaign Reports
are handoff artifacts written as files in the corresponding Agent Workspace
subdirectories unless an agent-safe wrapper command is explicitly added later.

The agent may run shell, Python, and common text/data tools for bounded
agent-side work such as editing Candidate Experiment files, parsing exposed
metrics, making Research Note figures, summarizing JSON/CSV artifacts,
performing static source inspection, and checking tensor-shape ideas on
synthetic toy inputs.

The agent must not run Candidate Experiments inside the VM. It must not run
Docker, GPU tools, cluster scheduler commands, direct training/evaluation scripts,
Harness execution backends, or any command intended to submit or execute a Run
outside the agent-safe wrapper. Authoritative Runs,
authoritative Results, Post-Run Evaluations, ledger ingestion, resource use,
and candidate execution happen only through Harness-owned processes outside the
Agent Control Boundary.

## Data access policy

Read-only `/data` inspection is allowed for hypothesis formation when the
Harness mounts approved Research Problem data. The agent may inspect schema,
metadata, filenames, small samples, class balance, or qualitative examples to
understand the Research Problem and write better proposals.

Candidate Experiment code must remain data-path agnostic. It must not hard-code
`/data` paths, implement candidate-owned data loading, probe mounted data from
model/helper code, cache dataset-derived side channels, or treat local analysis
as an official evaluation. Authoritative Results come from the Harness after it
validates and executes submitted Candidate Experiments.

## `agent-boundary.toml` schema

`ml-autoresearch prepare-agent-boundary` reads `agent-boundary.toml` from the
project root. A minimal configuration is:

```toml
[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true
```

The `[agent_control_boundary]` table accepts:

- `distro` — optional non-empty string; defaults to `"debian"`.
- `image` — optional non-empty string; defaults to
  `"../../containers/ml-autoresearch-agent"`.
- `allow_egress` — optional boolean; defaults to `true`.

The `image` value is copied into the generated pi-fort configuration. Relative
image paths are interpreted by pi-fort relative to the generated
`agent-work/.pi/fort.toml` file, not relative to the project root or the
`agent-work/` current working directory. Thus the default
`../../containers/ml-autoresearch-agent` resolves from `agent-work/.pi/` to the
project's `containers/ml-autoresearch-agent` image reference.

Approved read-only Research Problem data mounts are optional and use an array of
tables:

```toml
[[data_mounts]]
name = "example_research_problem_data"
path = "/path/to/research-problem-data"
target = "/data/example-research-problem-data"
readonly = true
```

Each `[[data_mounts]]` entry accepts:

- `name` — required non-empty string. If `target` is omitted, this name is used
  to derive `target = "/data/{name}"`.
- `path` — required non-empty string naming an existing host path. Relative
  paths are resolved relative to the project root.
- `target` — optional non-empty string; defaults to `/data/{name}`. The target
  must be a non-overlapping direct child of `/data`, for example
  `/data/example-research-problem-data`; nested targets such as
  `/data/example-research-problem-data/train` are rejected.
- `readonly` — optional boolean, but if present it must be `true`. Writable data
  mounts are rejected.

## Implementation using pi-fort

The `.pi/fort.toml` configuration when running Pi inside the Agent Control
Boundary should be generated from `agent-boundary.toml` by the Harness. A
representative generated configuration is:

```toml
# Generated by ml-autoresearch prepare-agent-boundary
enabled = true
allow_egress = true
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
mounts = [
  {path="../agent-reference", target="/reference", readonly=true},
  {path="../agent-history", target="/history", readonly=true},
  {path="../agent-history/candidates", target="/history/candidates", readonly=true},
  {path="../agent-history/runs", target="/history/runs", readonly=true},
  {path="../agent-history/batches", target="/history/batches", readonly=true},
  {path="../agent-history/research-notes", target="/history/research-notes", readonly=true},
  {path="../docs", target="/docs", readonly=true},
  {path="/path/to/research-problem-package", target="/research-problem", readonly=true},
  {path="/path/to/research-problem-data", target="/data/example-research-problem-data", readonly=true},
]
```

## Agent image and dependency boundary

The Agent Control Boundary image installs the base `ml-autoresearch` package and
exposes the `ml-autoresearch-agent` console script for observation and static
Candidate Experiment preparation. During boundary preparation, the current
Harness `src/ml_autoresearch` tree is mounted read-only over the image package
path so agent-safe validation commands use the same Candidate Experiment Contract
implementation as the outer Harness. The image intentionally does not install PyTorch,
NVIDIA/CUDA libraries, Docker tooling, GPU utilities, or Run-execution
dependencies. These heavy ML/runtime dependencies belong to the outer Harness and Candidate Execution Boundary, where Candidate Experiments are validated,
smoke-tested, trained, and evaluated under Harness-owned policy.

The agent image build smoke-checks `ml-autoresearch-agent --help` plus allowed
static commands such as `list-runs`, `validate-candidate`, and
`prepare-candidate-submission`. The separate runner image remains responsible
for the PyTorch/CUDA stack used by Candidate Execution Boundary training.

## Setup

The host-side Harness prepares the Agent Reference Snapshot, Research History
snapshot, Agent Workspace directory layout, and managed pi-fort configuration
from root `agent-boundary.toml`:

```shell
ml-autoresearch prepare-agent-boundary
```

The command overwrites only managed pi-fort files under
`agent-work/.pi/fort.toml` and `agent-work/.pi/fort.d/`, refreshes managed
Autoresearch Skill Set files under `agent-work/.pi/skills/`, and rewrites the
managed workspace instruction file `agent-work/AGENTS.md`; it does not delete
the whole `agent-work/.pi` directory or existing Agent Workspace outputs.

```shell
pi install -l git:git@github.com:ian-ross/pi-fort
cd agent-work
pi
```
