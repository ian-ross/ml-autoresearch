"""Candidate Experiment contract validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class CandidateValidationError(ValueError):
    """Raised when a Candidate Experiment does not satisfy the v1 contract."""


class TrainingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: Literal["bce_dice"]
    optimizer: Literal["adamw"]
    learning_rate: float = Field(ge=1e-5, le=3e-3)
    batch_size: int = Field(ge=1, le=32)
    max_epochs: int = Field(ge=1, le=100)


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sampling_policy: Literal["sequential", "deterministic_shuffle"] = "sequential"


class AuxiliaryTargetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["line"]
    output: Literal["line_logits"]
    loss: Literal["weighted_bce"]
    weight: float = Field(ge=0.0, le=1.0)


class CandidateManifest(BaseModel):
    """Normalized v1 Candidate Experiment source manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    input_mode: Literal["single_frame_rgb"]
    output_form: Literal["mask_logits"]
    auxiliary_targets: list[AuxiliaryTargetManifest] = Field(default_factory=list)
    data: DataManifest = Field(default_factory=DataManifest)
    training: TrainingManifest


_ALLOWED_FILENAMES = {"manifest.yaml", "model.py", "README.md"}
_ALLOWED_SUFFIXES = {".py"}
_FORBIDDEN_SUFFIXES = {
    ".sh",
    ".ipynb",
    ".pt",
    ".pth",
    ".ckpt",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
    ".npy",
    ".npz",
}


def validate_candidate_directory(candidate_dir: str | Path) -> CandidateManifest:
    """Validate a local Candidate Experiment directory and return its manifest.

    Issue 1 validates only the source contract. It does not import or execute
    candidate Python code.
    """

    path = Path(candidate_dir)
    if not path.exists():
        raise CandidateValidationError(f"candidate directory does not exist: {path}")
    if not path.is_dir():
        raise CandidateValidationError(f"candidate source must be a directory: {path}")

    _validate_required_files(path)
    _validate_file_allowlist(path)
    return _load_manifest(path / "manifest.yaml")


def _validate_required_files(path: Path) -> None:
    missing = [name for name in ("manifest.yaml", "model.py") if not (path / name).is_file()]
    if missing:
        raise CandidateValidationError("missing required candidate files: " + ", ".join(missing))


def _validate_file_allowlist(path: Path) -> None:
    for item in path.rglob("*"):
        relative = item.relative_to(path)
        if item.is_symlink():
            raise CandidateValidationError(f"symlink is forbidden in candidate source: {relative}")
        if item.is_dir():
            if item.name.startswith("."):
                raise CandidateValidationError(f"hidden directory is forbidden in candidate source: {relative}")
            continue
        if item.name.startswith("."):
            raise CandidateValidationError(f"hidden file is forbidden in candidate source: {relative}")
        if item.name in _ALLOWED_FILENAMES:
            continue
        if item.suffix in _FORBIDDEN_SUFFIXES:
            raise CandidateValidationError(f"forbidden candidate file: {relative}")
        if item.suffix in _ALLOWED_SUFFIXES:
            continue
        raise CandidateValidationError(f"forbidden candidate file: {relative}")


def _load_manifest(path: Path) -> CandidateManifest:
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise CandidateValidationError(f"invalid manifest.yaml: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise CandidateValidationError("manifest.yaml must contain a mapping")

    try:
        return CandidateManifest.model_validate(loaded)
    except ValidationError as exc:
        details = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            input_value = error.get("input")
            input_detail = f" (got {input_value!r})" if input_value is not None else ""
            details.append(f"{loc}: {error['msg']}{input_detail}")
        raise CandidateValidationError("invalid manifest.yaml: " + "; ".join(details)) from exc
