"""Lightweight shared Harness exceptions."""

from __future__ import annotations


class SmokeTestError(RuntimeError):
    """Raised when a Candidate Experiment fails synthetic model smoke testing."""


class TrainingError(RuntimeError):
    """Raised when Harness-owned training fails."""


class ResearchProblemDataError(ValueError):
    """Raised when a configured Research Problem data root is missing or malformed."""
