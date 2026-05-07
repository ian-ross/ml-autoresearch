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

## Decision

Keep exploring, abandon, modify, or rerun?

## Next proposed change

What should the next Experiment Proposal test?
```
