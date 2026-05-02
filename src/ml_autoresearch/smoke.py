"""Synthetic PyTorch smoke tests for copied Candidate Experiments."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator

import torch
from torch import nn


INPUT_SPEC = {"mode": "single_frame_rgb", "shape": [3, 128, 128]}
OUTPUT_SPEC = {"form": "mask_logits", "shape": [1, 128, 128]}
SYNTHETIC_BATCH_SHAPE = [2, 3, 128, 128]
SYNTHETIC_TARGET_SHAPE = [2, 1, 128, 128]
MAX_PARAMETER_COUNT = 10_000_000


class SmokeTestError(RuntimeError):
    """Raised when a Candidate Experiment fails synthetic model smoke testing."""


@dataclass(frozen=True)
class SmokeTestResult:
    parameter_count: int
    input_spec: dict[str, object]
    output_spec: dict[str, object]


def smoke_test_run(run_dir: str | Path) -> SmokeTestResult:
    """Import copied candidate/model.py and run a cheap synthetic PyTorch check."""

    path = Path(run_dir)
    return smoke_test_candidate(path / "candidate", path / "outputs")


def smoke_test_candidate(candidate_dir: str | Path, outputs_dir: str | Path) -> SmokeTestResult:
    """Import a Candidate Experiment and write smoke-test outputs under outputs_dir."""

    candidate_dir = Path(candidate_dir)
    outputs_dir = Path(outputs_dir)
    log_path = outputs_dir / "logs" / "smoke_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["Starting synthetic PyTorch smoke test."]
    try:
        module = _import_candidate_model(candidate_dir)
        build_model = getattr(module, "build_model", None)
        if not callable(build_model):
            raise SmokeTestError("missing required build_model(input_spec, output_spec)")

        try:
            model = build_model(dict(INPUT_SPEC), dict(OUTPUT_SPEC))
        except Exception as exc:  # noqa: BLE001 - convert candidate exceptions to smoke failures.
            raise SmokeTestError(f"build_model failed: {exc}") from exc

        if not isinstance(model, nn.Module):
            raise SmokeTestError("build_model must return a torch.nn.Module")

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > MAX_PARAMETER_COUNT:
            raise SmokeTestError(
                f"parameter-budget violation: {parameter_count} parameters exceeds limit {MAX_PARAMETER_COUNT}"
            )

        model.train()
        inputs = torch.zeros(SYNTHETIC_BATCH_SHAPE, dtype=torch.float32)
        target = torch.zeros(SYNTHETIC_TARGET_SHAPE, dtype=torch.float32)

        try:
            raw_output = model(inputs)
        except Exception as exc:  # noqa: BLE001
            raise SmokeTestError(f"forward pass failed: {exc}") from exc

        mask_logits, output_names = _extract_mask_logits(raw_output)
        _validate_mask_logits(mask_logits)

        try:
            loss = torch.nn.functional.mse_loss(mask_logits, target)
            loss.backward()
        except Exception as exc:  # noqa: BLE001
            raise SmokeTestError(f"backward pass failed: {exc}") from exc

        summary = {
            "parameter_count": parameter_count,
            "parameter_budget": {"max_parameters": MAX_PARAMETER_COUNT},
            "input_spec": INPUT_SPEC,
            "output_spec": OUTPUT_SPEC,
            "synthetic_batch_shape": SYNTHETIC_BATCH_SHAPE,
            "synthetic_target_shape": SYNTHETIC_TARGET_SHAPE,
            "output": {
                "names": output_names,
                "shape": list(mask_logits.shape),
                "dtype": str(mask_logits.dtype),
            },
        }
        (outputs_dir / "model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        lines.append(f"Parameter count: {parameter_count}")
        lines.append("Smoke test accepted.")
        log_path.write_text("\n".join(lines) + "\n")
        return SmokeTestResult(parameter_count, dict(INPUT_SPEC), dict(OUTPUT_SPEC))
    except SmokeTestError as exc:
        lines.append(f"Smoke test failed: {exc}")
        log_path.write_text("\n".join(lines) + "\n")
        raise
    except Exception as exc:  # noqa: BLE001 - record unexpected harness/import failures with trace.
        reason = f"candidate import or smoke test failed: {exc}"
        lines.append(f"Smoke test failed: {reason}")
        lines.append(traceback.format_exc())
        log_path.write_text("\n".join(lines) + "\n")
        raise SmokeTestError(reason) from exc


def _extract_mask_logits(raw_output: object) -> tuple[torch.Tensor, list[str]]:
    if isinstance(raw_output, torch.Tensor):
        return raw_output, ["mask_logits"]
    if isinstance(raw_output, dict):
        keys = sorted(str(key) for key in raw_output.keys())
        if keys != ["mask_logits"]:
            raise SmokeTestError(f"unexpected output keys: {keys}; expected ['mask_logits']")
        value = raw_output["mask_logits"]
        if not isinstance(value, torch.Tensor):
            raise SmokeTestError("output 'mask_logits' must be a torch.Tensor")
        return value, ["mask_logits"]
    raise SmokeTestError("model output must be Tensor or {'mask_logits': Tensor}")


def _validate_mask_logits(mask_logits: torch.Tensor) -> None:
    expected_shape = tuple(SYNTHETIC_TARGET_SHAPE)
    if tuple(mask_logits.shape) != expected_shape:
        raise SmokeTestError(f"bad output shape: {list(mask_logits.shape)}; expected {list(expected_shape)}")
    if not torch.is_floating_point(mask_logits):
        raise SmokeTestError(f"bad output dtype: {mask_logits.dtype}; expected floating point mask logits")


def _import_candidate_model(candidate_dir: Path) -> ModuleType:
    model_path = candidate_dir / "model.py"
    if not model_path.is_file():
        raise SmokeTestError(f"copied candidate model.py not found: {model_path}")
    module_name = f"_ml_autoresearch_candidate_{candidate_dir.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise SmokeTestError(f"could not create import spec for {model_path}")
    module = importlib.util.module_from_spec(spec)
    with _candidate_import_path(candidate_dir):
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            raise SmokeTestError(f"candidate model import failed: {exc}") from exc
        finally:
            sys.modules.pop(module_name, None)
    return module


@contextmanager
def _candidate_import_path(candidate_dir: Path) -> Iterator[None]:
    candidate_path = str(candidate_dir.resolve())
    original_path = list(sys.path)
    sys.path.insert(0, candidate_path)
    try:
        yield
    finally:
        sys.path[:] = original_path
