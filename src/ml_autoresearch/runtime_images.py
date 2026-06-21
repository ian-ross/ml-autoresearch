"""Runtime Image Build and validation support."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable, Sequence

import tomllib

from ml_autoresearch.package_resources import stage_workspace_container_build_recipes
from ml_autoresearch.workspace import WORKSPACE_CONFIG_FILENAME, workspace_config_path

RUNTIME_STATE_DIR = ".ml-autoresearch"
AGENT_IMAGE_RELATIVE = Path(RUNTIME_STATE_DIR) / "images" / "agent"
RUNNER_IMAGE_RELATIVE = Path(RUNTIME_STATE_DIR) / "images" / "runner"
VALIDATION_STAMP_RELATIVE = Path(RUNTIME_STATE_DIR) / "runtime-images.validated.json"
DEV_OVERRIDE_ENV = "ML_AUTORESEARCH_RUNTIME_IMAGE_SOURCE_OVERRIDE"


class RuntimeImageError(ValueError):
    """Raised when runtime image build or validation cannot continue."""


@dataclass(frozen=True)
class RuntimeImageBuildResult:
    """Summary of a Runtime Image Build operation."""

    workspace_root: Path
    runner_image_tag: str
    agent_image_path: Path
    recipes_path: Path
    harness_identity: dict[str, object]
    config_updated: bool
    executed_commands: tuple[tuple[str, ...], ...]

    def model_dump(self) -> dict[str, object]:
        return {
            "workspace_root": str(self.workspace_root),
            "runner_image_tag": self.runner_image_tag,
            "agent_image_path": str(self.agent_image_path),
            "recipes_path": str(self.recipes_path),
            "harness_identity": self.harness_identity,
            "config_updated": self.config_updated,
            "executed_commands": [list(command) for command in self.executed_commands],
        }


def build_runtime_images(
    workspace_root: str | Path = Path("."),
    *,
    update_config: bool = False,
    execute: bool = True,
    command_runner: Callable[[list[str]], object] | None = None,
) -> RuntimeImageBuildResult:
    """Build or prepare workspace-specific runtime image assets from packaged recipes."""

    root = Path(workspace_root).resolve()
    _require_workspace_config(root)
    identity = current_harness_identity(root)
    runner_tag = default_runner_image_tag(root, identity)
    agent_path = root / AGENT_IMAGE_RELATIVE
    recipes = stage_workspace_container_build_recipes(root).destination

    commands = [
        ["docker", "build", "-f", str(recipes / "Dockerfile.runner"), "-t", runner_tag, str(root)],
        ["docker", "build", "-f", str(recipes / "Dockerfile.agent"), "-t", default_agent_oci_image_tag(root, identity), str(root)],
        [
            "gondolin",
            "build",
            "--config",
            str(recipes / "gondolin-build-config.json"),
            "--output",
            str(agent_path),
        ],
    ]
    executed: list[tuple[str, ...]] = []
    if execute:
        runner = command_runner or _run_command
        for command in commands:
            runner(command)
            executed.append(tuple(command))

    _write_runtime_metadata(root / RUNNER_IMAGE_RELATIVE, "runner", identity, {"tag": runner_tag})
    _write_runtime_metadata(agent_path, "agent", identity, {"path": str(agent_path)})
    if update_config:
        _update_workspace_config(root, runner_tag=runner_tag, agent_image_path=agent_path)
    return RuntimeImageBuildResult(
        workspace_root=root,
        runner_image_tag=runner_tag,
        agent_image_path=agent_path,
        recipes_path=recipes,
        harness_identity=identity,
        config_updated=update_config,
        executed_commands=tuple(executed),
    )


def validate_runtime_images(workspace_root: str | Path = Path(".")) -> dict[str, object]:
    """Validate configured runtime images and write the Runtime Image Validation Stamp."""

    root = Path(workspace_root).resolve()
    _require_workspace_config(root)
    config = _load_workspace_toml(root)
    identity = current_harness_identity(root)
    runner_tag = _configured_runner_tag(config) or default_runner_image_tag(root, identity)
    agent_path = _configured_agent_path(config, root) or root / AGENT_IMAGE_RELATIVE

    runner_metadata = _read_runtime_metadata(root / RUNNER_IMAGE_RELATIVE, "Candidate Execution Boundary runner image")
    agent_metadata = _read_runtime_metadata(agent_path, "Agent Runtime Image asset")
    _require_metadata_matches(runner_metadata, expected_kind="runner", expected_identity=identity)
    _require_metadata_matches(agent_metadata, expected_kind="agent", expected_identity=identity)
    if runner_metadata.get("tag") != runner_tag:
        raise RuntimeImageError(
            f"configured runner image {runner_tag!r} does not match built runner image {runner_metadata.get('tag')!r}"
        )
    if Path(str(agent_metadata.get("path", ""))).resolve() != agent_path.resolve():
        raise RuntimeImageError(
            f"configured Agent Runtime Image path {agent_path} does not match built asset {agent_metadata.get('path')}"
        )

    stamp = {
        "harness_identity": identity,
        "image_identity": {
            "runner": {"tag": runner_tag, "metadata_path": str(root / RUNNER_IMAGE_RELATIVE / "runtime-image.json")},
            "agent": {"path": str(agent_path), "metadata_path": str(agent_path / "runtime-image.json")},
        },
        "dev_override": _dev_override_state(root),
        "workspace_config": _workspace_config_identity(root),
        "validated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    stamp_path = root / VALIDATION_STAMP_RELATIVE
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    return stamp


def require_runtime_image_validation(workspace_root: str | Path = Path(".")) -> dict[str, object]:
    """Require a fresh Runtime Image Validation Stamp for runtime-image operations."""

    root = Path(workspace_root).resolve()
    _require_workspace_config(root)
    stamp_path = root / VALIDATION_STAMP_RELATIVE
    if not stamp_path.is_file():
        raise RuntimeImageError(_validation_failure_message(root, "Runtime Image Validation Stamp is missing"))
    try:
        stamp = json.loads(stamp_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeImageError(_validation_failure_message(root, f"Runtime Image Validation Stamp is invalid JSON: {exc}")) from exc
    if not isinstance(stamp, dict):
        raise RuntimeImageError(_validation_failure_message(root, "Runtime Image Validation Stamp must be a JSON object"))

    config = _load_workspace_toml(root)
    identity = current_harness_identity(root)
    runner_tag = _configured_runner_tag(config) or default_runner_image_tag(root, identity)
    agent_path = _configured_agent_path(config, root) or root / AGENT_IMAGE_RELATIVE
    expected = {
        "harness_identity": identity,
        "image_identity": {
            "runner": {"tag": runner_tag, "metadata_path": str(root / RUNNER_IMAGE_RELATIVE / "runtime-image.json")},
            "agent": {"path": str(agent_path), "metadata_path": str(agent_path / "runtime-image.json")},
        },
        "dev_override": _dev_override_state(root),
        "workspace_config": _workspace_config_identity(root),
    }
    for key, expected_value in expected.items():
        if stamp.get(key) != expected_value:
            raise RuntimeImageError(_validation_failure_message(root, f"Runtime Image Validation Stamp is stale or mismatched for {key}"))
    return stamp


def runtime_image_validation_skip_warning(command_name: str, workspace_root: str | Path = Path(".")) -> str:
    root = Path(workspace_root).resolve()
    return (
        f"WARNING: --skip-runtime-image-validation used for {command_name}; "
        f"runtime images were not checked against {root / WORKSPACE_CONFIG_FILENAME}, Harness identity, image identity, or development source override."
    )


def current_harness_identity(workspace_root: str | Path = Path(".")) -> dict[str, object]:
    """Return the package or explicit development-source identity for runtime image validation."""

    root = Path(workspace_root).resolve()
    override = _configured_dev_override(root)
    if override is not None:
        source_path, source = override
        return _source_identity(source_path, source=source)
    return {"kind": "package", "name": "ml-autoresearch", "version": metadata.version("ml-autoresearch")}


def default_runner_image_tag(workspace_root: str | Path, identity: dict[str, object] | None = None) -> str:
    """Return the workspace- and Harness-specific default runner Docker tag."""

    root = Path(workspace_root).resolve()
    identity = identity or current_harness_identity(root)
    identity_part = str(identity.get("version") or identity.get("fingerprint") or "unknown")
    workspace_part = _slug(root.name or "workspace", max_length=36)
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    tag = _slug(f"{workspace_part}-{identity_part}-{digest}", max_length=120)
    return f"ml-autoresearch-runner:{tag}"


def default_agent_oci_image_tag(workspace_root: str | Path, identity: dict[str, object] | None = None) -> str:
    root = Path(workspace_root).resolve()
    identity = identity or current_harness_identity(root)
    identity_part = str(identity.get("version") or identity.get("fingerprint") or "unknown")
    return f"ml-autoresearch-agent:{_slug(f'{root.name}-{identity_part}', max_length=120)}"


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _require_workspace_config(root: Path) -> None:
    path = workspace_config_path(root)
    if not path.is_file():
        raise RuntimeImageError(f"missing Workspace Configuration: {path}")


def _load_workspace_toml(root: Path) -> dict[str, object]:
    path = root / WORKSPACE_CONFIG_FILENAME
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeImageError(f"invalid Workspace Configuration {path}: {exc}") from exc


def _workspace_config_identity(root: Path) -> dict[str, object]:
    path = root / WORKSPACE_CONFIG_FILENAME
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _validation_failure_message(root: Path, reason: str) -> str:
    return (
        f"{reason}. Run `ml-autoresearch build-runtime-images --workspace-root {root} --update-config` "
        f"if needed, then `ml-autoresearch validate-runtime-images --workspace-root {root}`. "
        "Advanced operators may bypass with --skip-runtime-image-validation."
    )


def _configured_runner_tag(config: dict[str, object]) -> str | None:
    candidate_execution = config.get("candidate_execution", {})
    if isinstance(candidate_execution, dict) and isinstance(candidate_execution.get("docker_image"), str):
        return str(candidate_execution["docker_image"])
    runtime_images = config.get("runtime_images", {})
    if isinstance(runtime_images, dict) and isinstance(runtime_images.get("runner_image"), str):
        return str(runtime_images["runner_image"])
    return None


def _configured_agent_path(config: dict[str, object], root: Path) -> Path | None:
    settings = config.get("agent_control_boundary", {})
    if not isinstance(settings, dict) or not isinstance(settings.get("image"), str):
        return None
    path = Path(str(settings["image"])).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _configured_dev_override(root: Path) -> tuple[Path, str] | None:
    env = os.environ.get(DEV_OVERRIDE_ENV)
    if env:
        return Path(env).expanduser().resolve(), "environment"
    try:
        runtime_images = _load_workspace_toml(root).get("runtime_images", {})
    except RuntimeImageError:
        return None
    if isinstance(runtime_images, dict) and isinstance(runtime_images.get("dev_source_path"), str):
        return Path(str(runtime_images["dev_source_path"])).expanduser().resolve(), "config"
    return None


def _dev_override_state(root: Path) -> dict[str, object]:
    override = _configured_dev_override(root)
    if override is None:
        return {"enabled": False}
    source_path, source = override
    return {"enabled": True, "source": source, "path": str(source_path)}


def _source_identity(source_path: Path, *, source: str) -> dict[str, object]:
    if not source_path.exists() or not source_path.is_dir():
        raise RuntimeImageError(f"development source override path does not exist or is not a directory: {source_path}")
    git_commit = _git_output(source_path, ["rev-parse", "HEAD"])
    git_status = _git_output(source_path, ["status", "--porcelain"])
    if git_commit:
        state = "dirty" if git_status else "clean"
        fingerprint = hashlib.sha256(f"{source_path}\0{git_commit}\0{git_status}".encode()).hexdigest()[:16]
        return {
            "kind": "source",
            "path": str(source_path),
            "source": source,
            "git_commit": git_commit,
            "git_state": state,
            "fingerprint": fingerprint,
        }
    fingerprint = hashlib.sha256(str(source_path).encode()).hexdigest()[:16]
    return {"kind": "source", "path": str(source_path), "source": source, "git_state": "not-a-git-worktree", "fingerprint": fingerprint}


def _git_output(cwd: Path, args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(["git", *args], cwd=cwd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _write_runtime_metadata(directory: Path, kind: str, identity: dict[str, object], image_data: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"kind": kind, "harness_identity": identity, **image_data}
    (directory / "runtime-image.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_runtime_metadata(directory: Path, label: str) -> dict[str, object]:
    path = directory / "runtime-image.json"
    if not path.is_file():
        raise RuntimeImageError(f"{label} metadata is missing: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeImageError(f"invalid {label} metadata {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeImageError(f"invalid {label} metadata {path}: expected object")
    return data


def _require_metadata_matches(metadata_payload: dict[str, object], *, expected_kind: str, expected_identity: dict[str, object]) -> None:
    if metadata_payload.get("kind") != expected_kind:
        raise RuntimeImageError(f"runtime image metadata kind mismatch: expected {expected_kind}")
    if metadata_payload.get("harness_identity") != expected_identity:
        raise RuntimeImageError("Harness identity mismatch for runtime image metadata")


def _update_workspace_config(root: Path, *, runner_tag: str, agent_image_path: Path) -> None:
    path = root / WORKSPACE_CONFIG_FILENAME
    text = path.read_text()
    text = _set_toml_key(text, "candidate_execution", "docker_image", runner_tag)
    text = _set_toml_key(text, "agent_control_boundary", "image", str(agent_image_path))
    path.write_text(text)


def _set_toml_key(text: str, section: str, key: str, value: str) -> str:
    lines = text.splitlines()
    section_header = f"[{section}]"
    key_line = f'{key} = "{value}"'
    section_start = next((index for index, line in enumerate(lines) if line.strip() == section_header), None)
    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([section_header, key_line])
        return "\n".join(lines) + "\n"
    next_section = len(lines)
    for index in range(section_start + 1, len(lines)):
        if lines[index].lstrip().startswith("["):
            next_section = index
            break
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(section_start + 1, next_section):
        if key_re.match(lines[index]):
            lines[index] = key_line
            return "\n".join(lines) + "\n"
    lines.insert(next_section, key_line)
    return "\n".join(lines) + "\n"


def _slug(value: str, *, max_length: int) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip(".-")
    slug = re.sub(r"-+", "-", slug) or "workspace"
    return slug[:max_length].strip(".-") or "workspace"
