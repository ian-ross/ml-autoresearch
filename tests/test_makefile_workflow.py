import subprocess
from pathlib import Path


DEFAULT_RUNNER_IMAGE = "ml-autoresearch-runner:local"
MANUAL_BUILD_COMMAND = f"docker build -t {DEFAULT_RUNNER_IMAGE} ."


def test_makefile_builds_default_local_runner_image() -> None:
    assert Path("Makefile").exists()

    result = subprocess.run(
        ["make", "--dry-run", "runner-image"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert MANUAL_BUILD_COMMAND in result.stdout


def test_user_docs_prefer_makefile_runner_image_workflow() -> None:
    readme = Path("README.md").read_text()
    dependency_strategy = Path("docs/dependency-strategy.md").read_text()

    for text in (readme, dependency_strategy):
        assert "make runner-image" in text
        assert DEFAULT_RUNNER_IMAGE in text
        assert MANUAL_BUILD_COMMAND in text
