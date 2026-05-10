# Research Notes

Research Notes are lightweight Markdown records for Human-Guided Research Iterations. They are the current input layer for the Research Loop: humans and agents review prior Results, constraints, and notes before writing the next Experiment Proposal.

When writing a new Research Note, also update the top-level Experiment Index at `EXPERIMENT_INDEX.md`.

A Research Note may describe one Run or compare multiple Runs. It should reference Run IDs and local artifact paths rather than duplicating raw logs or metrics.

Future richer reporting, such as LaTeX experiment reports and current-status summaries, may be generated from or informed by these notes.

## Suggested note structure

```markdown
# YYYY-MM-DD short-title

## Hypothesis

What was expected to improve, and why?

## Candidate Experiment(s)

- Candidate Experiment path or ID:
- Relevant Experiment Proposal, if any:

## Run(s)

- Run ID:
- Dataset mode/subset:
- Harness/backend notes:

## Key metrics

| Run | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | ---: | ---: | ---: | ---: | ---: |

## Qualitative observations

Notes from `prediction_samples/`, including obvious false positives, false negatives, mask shape issues, or artifact problems.

## Research Figures

Optional Research Figures that illustrate the observations. Use the provenance format below.

## Decision

Keep exploring, abandon, modify, or rerun?

## Next proposed change

What should the next Experiment Proposal test?
```

## Research Figure provenance

Research Figures are images embedded in or linked from a Research Note to support qualitative interpretation. A figure may be copied into the note directory when that improves readability, but existing Harness artifacts should normally be referenced in place when appropriate. In particular, saved `outputs/prediction_samples/` artifacts and Post-Run Evaluation artifacts can be linked without copying as long as the source artifact path remains traceable.

Each Research Figure must have a matching fenced `research-figures` YAML block. The Harness checker `validate_research_note_figure_provenance` validates these blocks for required provenance fields:

- `figure_id`: note-local stable identifier.
- exactly one of `source_run_id` or `source_evaluation_id`.
- `source_artifact_path`: artifact path relative to the source Run or Evaluation when possible.
- `reason`: why this figure was selected for the Research Note.

Optional Markdown links or image embeds may appear next to the prose; the fenced block is the machine-checkable provenance record.

### Single-Run note snippet

````markdown
## Qualitative observations

![False negative overlay](../runs/run_20260510_001/outputs/prediction_samples/sample_000_overlay.png)

The selected overlay shows the clearest missed thin contrail among the saved samples.

## Research Figures

```research-figures
figures:
  - figure_id: fig-overlay-001
    source_run_id: run_20260510_001
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows the clearest false negative in the saved prediction samples.
```
````

### Comparison note snippet

````markdown
## Qualitative observations

[Threshold sweep comparison](../evaluations/eval_20260510_threshold_sweep/threshold_sweep.png) shows that Run B improves Dice only at lower thresholds.

## Research Figures

```research-figures
figures:
  - figure_id: fig-threshold-sweep
    source_evaluation_id: eval_20260510_threshold_sweep
    source_artifact_path: evaluations/eval_20260510_threshold_sweep/threshold_sweep.png
    reason: Compares threshold sensitivity for the two Runs before choosing the next Candidate Experiment.
```
````

## Research Ledger event

When recording that a Research Note was written, use `record-research-event` or `record_research_event` with `event_type=research_note_written`. The event must include `note_path` and may include either:

- `figure_provenance_path`, pointing to the note section or sidecar file containing figure provenance; or
- `figure_provenance`, embedding the validated figure provenance metadata.

Example using a path reference:

```bash
ml-autoresearch record-research-event \
  --event-type research_note_written \
  --field note_path=research-notes/2026-05-10-example.md \
  --field figure_provenance_path=research-notes/2026-05-10-example.md#research-figures
```
