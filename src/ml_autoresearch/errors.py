"""Lightweight shared Harness exceptions."""

from __future__ import annotations


HARNESS_FAILURE_MARKER = "ML_AUTORESEARCH_FAILURE_CLASSIFICATION=harness_failure"


class HarnessBootstrapError(RuntimeError):
    """Raised when trusted image, provider, or data setup prevents Candidate execution."""

    failure_classification = "harness_failure"


class SmokeTestError(RuntimeError):
    """Raised when a Candidate Experiment fails synthetic model smoke testing."""


class TrainingError(RuntimeError):
    """Raised when Harness-owned training fails."""


class ResearchProblemDataError(ValueError):
    """Raised when a configured Research Problem data root is missing or malformed."""
