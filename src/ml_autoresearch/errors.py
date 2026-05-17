"""Lightweight shared Harness exceptions."""

from __future__ import annotations


class SmokeTestError(RuntimeError):
    """Raised when a Candidate Experiment fails synthetic model smoke testing."""


class TrainingError(RuntimeError):
    """Raised when Harness-owned training fails."""


class GVCCSDataError(ValueError):
    """Raised when a local GVCCS Dataset root is missing or malformed."""
