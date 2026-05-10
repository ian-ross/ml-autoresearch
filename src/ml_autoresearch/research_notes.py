"""Research Note helpers owned by the Harness."""

from __future__ import annotations

import re
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ResearchNoteFigureError(ValueError):
    """Raised when Research Figure provenance is malformed."""


class ResearchFigureProvenance(BaseModel):
    """Traceability metadata for one Research Figure in a Research Note."""

    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(min_length=1)
    source_run_id: str | None = Field(default=None, min_length=1)
    source_evaluation_id: str | None = Field(default=None, min_length=1)
    source_artifact_path: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_one_source_identifier(self) -> "ResearchFigureProvenance":
        source_count = sum(value is not None for value in (self.source_run_id, self.source_evaluation_id))
        if source_count != 1:
            raise ValueError("Research Figure provenance must include exactly one of source_run_id or source_evaluation_id")
        return self


class _ResearchFiguresBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figures: list[ResearchFigureProvenance]


_RESEARCH_FIGURES_BLOCK = re.compile(r"```research-figures\s*\n(?P<body>.*?)\n```", re.DOTALL)


def validate_research_note_figure_provenance(markdown: str) -> list[dict[str, Any]]:
    """Validate and return Research Figure provenance blocks from a note.

    Research Notes without Research Figures are valid and return an empty list.
    Any fenced ``research-figures`` block must contain YAML with a top-level
    ``figures`` list, where each figure records its source Run or Evaluation,
    source artifact path, and selection reason.
    """

    validated: list[dict[str, Any]] = []
    for match in _RESEARCH_FIGURES_BLOCK.finditer(markdown):
        try:
            payload = yaml.safe_load(match.group("body"))
        except yaml.YAMLError as exc:
            raise ResearchNoteFigureError(f"invalid research-figures YAML: {exc}") from exc
        try:
            block = _ResearchFiguresBlock.model_validate(payload)
        except ValidationError as exc:
            raise ResearchNoteFigureError(str(exc)) from exc
        validated.extend(figure.model_dump(exclude_none=True) for figure in block.figures)
    return validated
