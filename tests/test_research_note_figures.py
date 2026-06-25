from pathlib import Path

import pytest

from ml_autoresearch.research_notes import ResearchNoteFigureError, validate_research_note_figure_provenance
from research_problem_helpers import gvccs_research_problem_root


SINGLE_RUN_NOTE = """
# 2026-05-10 example

## Research Figures

```research-figures
figures:
  - figure_id: fig-overlay-001
    source_run_id: run_001
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows the clearest false negative in the saved prediction samples.
```
"""


COMPARISON_NOTE = """
# 2026-05-10 comparison

## Research Figures

```research-figures
figures:
  - figure_id: fig-threshold-sweep
    source_evaluation_id: eval_20260510_threshold_sweep
    source_artifact_path: evaluations/eval_20260510_threshold_sweep/threshold_sweep.png
    reason: Compares validation Dice sensitivity across two Runs.
```
"""


def test_validate_research_note_accepts_single_run_figure_provenance() -> None:
    figures = validate_research_note_figure_provenance(SINGLE_RUN_NOTE)

    assert figures == [
        {
            "figure_id": "fig-overlay-001",
            "source_run_id": "run_001",
            "source_artifact_path": "outputs/prediction_samples/sample_000_overlay.png",
            "reason": "Shows the clearest false negative in the saved prediction samples.",
        }
    ]


def test_validate_research_note_accepts_evaluation_figure_provenance() -> None:
    figures = validate_research_note_figure_provenance(COMPARISON_NOTE)

    assert figures[0]["source_evaluation_id"] == "eval_20260510_threshold_sweep"


def test_validate_research_note_rejects_figure_missing_required_provenance() -> None:
    note = """
## Research Figures

```research-figures
figures:
  - figure_id: fig-bad
    source_run_id: run_001
    reason: Missing artifact path.
```
"""

    with pytest.raises(ResearchNoteFigureError, match="source_artifact_path"):
        validate_research_note_figure_provenance(note)


def test_validate_research_note_requires_exactly_one_source_identifier() -> None:
    note = """
## Research Figures

```research-figures
figures:
  - figure_id: fig-bad
    source_run_id: run_001
    source_evaluation_id: eval_001
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Ambiguous source.
```
"""

    with pytest.raises(ResearchNoteFigureError, match="exactly one"):
        validate_research_note_figure_provenance(note)


def test_existing_research_notes_have_valid_figure_provenance_blocks() -> None:
    for note_path in (gvccs_research_problem_root() / "research-notes").glob("*.md"):
        validate_research_note_figure_provenance(note_path.read_text())


def test_research_note_documentation_describes_figure_provenance_format() -> None:
    text = (gvccs_research_problem_root() / "research-notes" / "README.md").read_text()

    assert "```research-figures" in text
    assert "source_run_id" in text
    assert "source_evaluation_id" in text
    assert "source_artifact_path" in text
    assert "reason" in text
    assert "outputs/prediction_samples" in text
    assert "Post-Run Evaluation" in text
    assert "record-research-event" in text
