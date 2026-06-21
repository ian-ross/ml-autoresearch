from __future__ import annotations

import json
from pathlib import Path

from conftest import invoke_typer_cli
from ml_autoresearch.cli import app
from ml_autoresearch.runtime_images import (
    RuntimeImageError,
    build_runtime_images,
    validate_runtime_images,
)


def _workspace_config(tmp_path: Path) -> Path:
    config = tmp_path / "ml-autoresearch.toml"
    config.write_text(
        """
[candidate_execution]
backend = "docker"

[agent_control_boundary]
distro = "debian"
""".lstrip()
    )
    return config


def test_build_runtime_images_stages_agent_assets_under_hidden_workspace_state_without_config_mutation(tmp_path: Path) -> None:
    config = _workspace_config(tmp_path)
    original = config.read_text()
    commands: list[list[str]] = []

    result = build_runtime_images(tmp_path, execute=False, command_runner=lambda command: commands.append(command))

    assert result.agent_image_path == tmp_path / ".ml-autoresearch" / "images" / "agent"
    assert (result.agent_image_path / "runtime-image.json").is_file()
    assert (tmp_path / ".ml-autoresearch" / "container-build-recipes" / "Dockerfile.runner").is_file()
    assert "0.1.0" in result.runner_image_tag
    assert result.config_updated is False
    assert config.read_text() == original
    assert commands == []


def test_build_runtime_images_update_config_records_workspace_specific_identities(tmp_path: Path) -> None:
    config = _workspace_config(tmp_path)

    result = build_runtime_images(tmp_path, execute=False, update_config=True)

    updated = config.read_text()
    assert f'docker_image = "{result.runner_image_tag}"' in updated
    assert f'image = "{result.agent_image_path}"' in updated
    assert "ml-autoresearch-runner:" in result.runner_image_tag
    assert tmp_path.name.lower()[:8] in result.runner_image_tag


def test_validate_runtime_images_writes_stamp_with_harness_and_image_identity(tmp_path: Path) -> None:
    _workspace_config(tmp_path)
    build = build_runtime_images(tmp_path, execute=False, update_config=True)

    stamp = validate_runtime_images(tmp_path)

    stamp_path = tmp_path / ".ml-autoresearch" / "runtime-images.validated.json"
    assert stamp_path.is_file()
    written = json.loads(stamp_path.read_text())
    assert written == stamp
    assert stamp["harness_identity"]["kind"] == "package"
    assert stamp["harness_identity"]["version"] == "0.1.0"
    assert stamp["image_identity"]["runner"]["tag"] == build.runner_image_tag
    assert stamp["image_identity"]["agent"]["path"] == str(build.agent_image_path)
    assert stamp["dev_override"]["enabled"] is False
    assert stamp["validated_at"].endswith("Z")


def test_validate_runtime_images_rejects_missing_images_or_assets(tmp_path: Path) -> None:
    _workspace_config(tmp_path)
    build_runtime_images(tmp_path, execute=False, update_config=True)
    (tmp_path / ".ml-autoresearch" / "images" / "agent" / "runtime-image.json").unlink()

    try:
        validate_runtime_images(tmp_path)
    except RuntimeImageError as exc:
        assert "Agent Runtime Image asset metadata is missing" in str(exc)
    else:
        raise AssertionError("expected missing agent asset failure")


def test_validate_runtime_images_rejects_version_mismatch(tmp_path: Path) -> None:
    _workspace_config(tmp_path)
    build_runtime_images(tmp_path, execute=False, update_config=True)
    runner_metadata = tmp_path / ".ml-autoresearch" / "images" / "runner" / "runtime-image.json"
    data = json.loads(runner_metadata.read_text())
    data["harness_identity"]["version"] = "9.9.9"
    runner_metadata.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    try:
        validate_runtime_images(tmp_path)
    except RuntimeImageError as exc:
        assert "Harness identity mismatch" in str(exc)
    else:
        raise AssertionError("expected version mismatch failure")


def test_dev_source_override_changes_identity_and_validation_metadata(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "harness-src"
    source.mkdir()
    (source / "marker.txt").write_text("dev\n")
    _workspace_config(tmp_path)
    monkeypatch.setenv("ML_AUTORESEARCH_RUNTIME_IMAGE_SOURCE_OVERRIDE", str(source))

    build_runtime_images(tmp_path, execute=False, update_config=True)
    stamp = validate_runtime_images(tmp_path)

    assert stamp["harness_identity"]["kind"] == "source"
    assert stamp["harness_identity"]["path"] == str(source.resolve())
    assert stamp["dev_override"]["enabled"] is True
    assert stamp["dev_override"]["source"] == "environment"


def test_runtime_image_cli_commands(tmp_path: Path) -> None:
    _workspace_config(tmp_path)

    build = invoke_typer_cli(app, ["build-runtime-images", "--no-execute", "--update-config"], cwd=tmp_path)
    assert build.returncode == 0, build.stderr
    validate = invoke_typer_cli(app, ["validate-runtime-images"], cwd=tmp_path)
    assert validate.returncode == 0, validate.stderr
    assert (tmp_path / ".ml-autoresearch" / "runtime-images.validated.json").is_file()
