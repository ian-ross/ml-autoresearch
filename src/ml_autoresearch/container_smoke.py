"""In-container smoke-test entrypoint for Docker-backed execution."""

from __future__ import annotations

from pathlib import Path

from ml_autoresearch.smoke import output_spec_from_resolved_manifest, smoke_test_candidate


def smoke_test_container(
    candidate_dir: Path = Path("/candidate"),
    outputs_dir: Path = Path("/outputs"),
    resolved_manifest_path: Path = Path("/resolved_manifest.yaml"),
) -> None:
    """Run container smoke testing using the mounted Run-scoped Resolved Manifest."""

    output_spec = output_spec_from_resolved_manifest(resolved_manifest_path)
    smoke_test_candidate(candidate_dir, outputs_dir, output_spec=output_spec)


def main() -> None:
    """Run the fixed Harness-owned smoke-test operation inside the container."""

    smoke_test_container()


if __name__ == "__main__":
    main()
