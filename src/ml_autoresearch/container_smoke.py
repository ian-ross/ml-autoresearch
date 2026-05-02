"""In-container smoke-test entrypoint for Docker-backed execution."""

from __future__ import annotations

from pathlib import Path

from ml_autoresearch.smoke import smoke_test_candidate


def main() -> None:
    """Run the fixed Harness-owned smoke-test operation inside the container."""

    smoke_test_candidate(Path("/candidate"), Path("/outputs"))


if __name__ == "__main__":
    main()
