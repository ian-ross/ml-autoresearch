import subprocess
from pathlib import Path


DEFAULT_RUNNER_IMAGE = "ml-autoresearch-runner:local"
DEFAULT_ENCLAVE_IMAGE = "ml-autoresearch-enclave:local"
RUNNER_BUILD_FRAGMENT = (
    f"docker build -f {Path.cwd()}/containers/Dockerfile.runner -t {DEFAULT_RUNNER_IMAGE} {Path.cwd()}"
)
ENCLAVE_BUILD_FRAGMENT = (
    f"docker build -f {Path.cwd()}/containers/Dockerfile.enclave -t {DEFAULT_ENCLAVE_IMAGE} {Path.cwd()}"
)


def test_container_makefile_builds_default_local_runner_image() -> None:
    assert Path("containers/Makefile").exists()

    result = subprocess.run(
        ["make", "-C", "containers", "--dry-run", "runner-image"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert RUNNER_BUILD_FRAGMENT in result.stdout


def test_container_makefile_builds_default_local_enclave_image() -> None:
    result = subprocess.run(
        ["make", "-C", "containers", "--dry-run", "enclave-image"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert ENCLAVE_BUILD_FRAGMENT in result.stdout


def test_user_docs_prefer_makefile_runner_image_workflow() -> None:
    readme = Path("README.md").read_text()
    dependency_strategy = Path("docs/dependency-strategy.md").read_text()

    for text in (readme, dependency_strategy):
        assert "make -C containers runner-image" in text
        assert DEFAULT_RUNNER_IMAGE in text
        assert "docker build -f containers/Dockerfile.runner -t ml-autoresearch-runner:local ." in text
