---
name: research-note-writer
description: Write Research Notes with figure provenance.
---

# Research Note Writer

## Use

Use after observing a Run, comparison, or Post-Run Evaluation to write a Research Note with metrics, qualitative observations, decisions, and Research Figures when useful.

## Read first

- `CONTEXT.md` for Research Note, Run, Result, and Candidate Experiment terms.
- `research-notes/README.md` for structure, research-figures YAML, and figure provenance requirements.
- `docs/run-lifecycle.md` for artifact paths.

## Instructions

Reference source Run IDs, Evaluation IDs, and artifact paths instead of copying raw logs. Every Research Figure needs a `research-figures` block with figure_id, exactly one source_run_id or source_evaluation_id, source_artifact_path, and reason. Record note creation through the ledger recorder.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness changes during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
