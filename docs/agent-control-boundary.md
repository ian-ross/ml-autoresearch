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
  - `/history/research-notes/` — prior Research Notes.
- `/docs` — trusted project documentation, including the Candidate Experiment
  Contract, Run lifecycle, request/report formats, and agent skill docs.
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
  {path="../agent-history/research-notes", target="/history/research-notes", readonly=true},
  {path="../docs", target="/docs", readonly=true},
  {path="/path/to/gvccs", target="/data/gvccs", readonly=true},
]
```

## Setup

The host-side Harness prepares the Agent Reference Snapshot, Research History
snapshot, Agent Workspace directory layout, and managed pi-fort configuration
from root `agent-boundary.toml`:

```shell
ml-autoresearch prepare-agent-boundary
```

The command overwrites only managed pi-fort files under
`agent-work/.pi/fort.toml` and `agent-work/.pi/fort.d/`; it does not delete the
whole `agent-work/.pi` directory or existing Agent Workspace outputs.

```shell
pi install -l git:git@github.com:ian-ross/pi-fort
cd agent-work
pi
```
