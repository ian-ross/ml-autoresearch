"""Reusable trusted frame-sequence helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

FRAME_TIMESTAMP_PATTERN = re.compile(r"(\d{14})")
DEFAULT_FRAME_SEQUENCE_STEP_SECONDS = 30


def timestamp_from_filename(filename: str) -> datetime | None:
    """Extract the first YYYYMMDDhhmmss timestamp from a filename."""

    match = FRAME_TIMESTAMP_PATTERN.search(filename)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def infer_timestamped_frame_sequences(
    items: Sequence[T],
    *,
    filename_for_item: Callable[[T], str | Path],
    step_seconds: int = DEFAULT_FRAME_SEQUENCE_STEP_SECONDS,
) -> list[list[T]]:
    """Infer Frame Sequences from timestamp-like item filenames.

    Items without parseable timestamps are ignored. Timestamped items are sorted
    by timestamp and filename; a new sequence starts when adjacent timestamps
    differ by more than ``step_seconds``.
    """

    timestamped = [(item, str(filename_for_item(item)), timestamp_from_filename(str(filename_for_item(item)))) for item in items]
    timestamped = [(item, filename, timestamp) for item, filename, timestamp in timestamped if timestamp is not None]
    timestamped.sort(key=lambda row: (row[2], row[1]))
    sequences: list[list[T]] = []
    previous_timestamp: datetime | None = None
    for item, _filename, timestamp in timestamped:
        if previous_timestamp is None or (timestamp - previous_timestamp).total_seconds() > step_seconds:
            sequences.append([])
        sequences[-1].append(item)
        previous_timestamp = timestamp
    return sequences
