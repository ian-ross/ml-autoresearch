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
import yaml

from ml_autoresearch.errors import SmokeTestError
from ml_autoresearch.research_problems import (
    ResearchProblemProviderConfig,
    ResearchProblemProviderLoadError,
    ResearchProblemSpecError,
    ResearchProblemSpecRegistry,
    load_research_problem_provider,
)


INPUT_SPEC = {"mode": "single_frame_rgb", "shape": [3, 128, 128]}
OUTPUT_SPEC = {"form": "mask_logits", "shape": [1, 128, 128]}
SYNTHETIC_BATCH_SHAPE = [2, 3, 128, 128]
SYNTHETIC_TARGET_SHAPE = [2, 1, 128, 128]
MAX_PARAMETER_COUNT = 10_000_000


@dataclass(frozen=True)
class SmokeTestResult:
    parameter_count: int
    input_spec: dict[str, object]
    output_spec: dict[str, object]


def smoke_test_run(run_dir: str | Path) -> SmokeTestResult:
    """Import copied candidate/model.py and run a cheap synthetic PyTorch check."""

    path = Path(run_dir)
    input_spec, output_spec = smoke_specs_from_resolved_manifest(path / "resolved_manifest.yaml")
    return smoke_test_candidate(path / "candidate", path / "outputs", input_spec=input_spec, output_spec=output_spec)


def smoke_test_candidate(
    candidate_dir: str | Path,
    outputs_dir: str | Path,
    *,
    input_spec: dict[str, object] | None = None,
    output_spec: dict[str, object] | None = None,
) -> SmokeTestResult:
    """Import a Candidate Experiment and write smoke-test outputs under outputs_dir."""

    candidate_dir = Path(candidate_dir)
    outputs_dir = Path(outputs_dir)
    input_spec = dict(input_spec or INPUT_SPEC)
    output_spec = dict(output_spec or OUTPUT_SPEC)
    synthetic_batch_shape = [2, *_spec_shape(input_spec, "input_spec")]
    synthetic_target_shape = [2, *_spec_shape(output_spec, "output_spec")]
    log_path = outputs_dir / "logs" / "smoke_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["Starting synthetic PyTorch smoke test."]
    try:
        module = _import_candidate_model(candidate_dir)
        build_model = getattr(module, "build_model", None)
        if not callable(build_model):
            raise SmokeTestError("missing required build_model(input_spec, output_spec)")

        try:
            model = build_model(dict(input_spec), dict(output_spec))
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
        inputs = torch.zeros(synthetic_batch_shape, dtype=torch.float32)
        target = torch.zeros(synthetic_target_shape, dtype=torch.float32)

        try:
            raw_output = model(inputs)
        except Exception as exc:  # noqa: BLE001
            raise SmokeTestError(f"forward pass failed: {exc}") from exc

        outputs = _extract_expected_outputs(raw_output, output_spec)
        primary_output_name = str(output_spec.get("form", "mask_logits"))
        mask_logits = outputs[primary_output_name]
        output_names = list(outputs.keys())

        try:
            loss = torch.nn.functional.mse_loss(mask_logits, target)
            loss.backward()
        except Exception as exc:  # noqa: BLE001
            raise SmokeTestError(f"backward pass failed: {exc}") from exc

        summary = {
            "parameter_count": parameter_count,
            "parameter_budget": {"max_parameters": MAX_PARAMETER_COUNT},
            "input_spec": input_spec,
            "output_spec": output_spec,
            "synthetic_batch_shape": synthetic_batch_shape,
            "synthetic_target_shape": synthetic_target_shape,
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
        return SmokeTestResult(parameter_count, dict(input_spec), dict(output_spec))
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


def smoke_specs_from_resolved_manifest(
    path: str | Path,
    *,
    research_problem_registry: ResearchProblemSpecRegistry | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = yaml.safe_load(Path(path).read_text())
    if not isinstance(manifest, dict):
        raise SmokeTestError("resolved_manifest.yaml must contain a mapping")
    try:
        spec = _research_problem_spec_for_resolved_manifest(manifest, research_problem_registry)
        return spec.build_input_spec(manifest), spec.build_output_spec(manifest)
    except ResearchProblemSpecError as exc:
        raise SmokeTestError(str(exc)) from exc


def input_spec_from_resolved_manifest(path: str | Path) -> dict[str, object]:
    return smoke_specs_from_resolved_manifest(path)[0]


def output_spec_from_resolved_manifest(path: str | Path) -> dict[str, object]:
    return smoke_specs_from_resolved_manifest(path)[1]


def _research_problem_spec_for_resolved_manifest(
    manifest: dict[str, object],
    registry: ResearchProblemSpecRegistry | None,
):
    research_problem = manifest.get("research_problem")
    if isinstance(research_problem, dict):
        spec_id = research_problem.get("id")
        if not isinstance(spec_id, str) or not spec_id:
            raise SmokeTestError("resolved manifest research_problem metadata is missing id")
    elif isinstance(research_problem, str):
        spec_id = research_problem
    else:
        raise SmokeTestError("resolved manifest must specify research_problem")
    if registry is not None:
        return registry.get(spec_id)
    loaded_registry = _registry_from_resolved_manifest_provider(manifest, spec_id)
    if loaded_registry is not None:
        return loaded_registry.get(spec_id)
    raise SmokeTestError("resolved manifest research_problem provider metadata is required")


def _registry_from_resolved_manifest_provider(
    manifest: dict[str, object], spec_id: str
) -> ResearchProblemSpecRegistry | None:
    research_problem = manifest.get("research_problem")
    if not isinstance(research_problem, dict):
        return None
    provider = research_problem.get("provider")
    if not isinstance(provider, dict):
        return None
    target = provider.get("target")
    package_root = provider.get("resolved_package_root")
    contract_version = research_problem.get("contract_version")
    if not isinstance(target, str) or not isinstance(package_root, str) or not isinstance(contract_version, str):
        return None
    registry = ResearchProblemSpecRegistry(active_id=spec_id)
    try:
        load_research_problem_provider(
            ResearchProblemProviderConfig(
                id=spec_id,
                package_root=Path(package_root),
                provider_target=target,
                expected_contract_version=contract_version,
            ),
            registry=registry,
        )
    except ResearchProblemProviderLoadError as exc:
        raise SmokeTestError(str(exc)) from exc
    return registry


def expected_output_names(output_spec: dict[str, object]) -> list[str]:
    auxiliary_outputs = output_spec.get("auxiliary_outputs", [])
    if not isinstance(auxiliary_outputs, list):
        raise SmokeTestError("output_spec auxiliary_outputs must be a list")
    return [str(output_spec.get("form", "mask_logits")), *[str(item["name"]) for item in auxiliary_outputs]]


def _expected_output_shapes(output_spec: dict[str, object]) -> dict[str, list[int]]:
    primary_name = str(output_spec.get("form", "mask_logits"))
    shapes = {primary_name: _spec_shape(output_spec, "output_spec")}
    auxiliary_outputs = output_spec.get("auxiliary_outputs", [])
    if not isinstance(auxiliary_outputs, list):
        raise SmokeTestError("output_spec auxiliary_outputs must be a list")
    for item in auxiliary_outputs:
        if not isinstance(item, dict):
            raise SmokeTestError("output_spec auxiliary_outputs must contain mappings")
        shapes[str(item["name"])] = _spec_shape(item, f"output_spec auxiliary output {item.get('name')!r}")
    return shapes


def _extract_expected_outputs(raw_output: object, output_spec: dict[str, object]) -> dict[str, torch.Tensor]:
    expected_names = expected_output_names(output_spec)
    expected_shapes = _expected_output_shapes(output_spec)
    if isinstance(raw_output, torch.Tensor):
        if len(expected_names) != 1:
            raise SmokeTestError("tensor output shorthand is only valid for mask-only candidates")
        outputs = {expected_names[0]: raw_output}
    elif isinstance(raw_output, dict):
        keys = sorted(str(key) for key in raw_output.keys())
        if keys != sorted(expected_names):
            raise SmokeTestError(f"unexpected output keys: {keys}; expected {sorted(expected_names)}")
        outputs = {}
        for name in expected_names:
            value = raw_output[name]
            if not isinstance(value, torch.Tensor):
                raise SmokeTestError(f"output '{name}' must be a torch.Tensor")
            outputs[name] = value
    else:
        raise SmokeTestError("model output must be Tensor or a dict of named tensors")
    for name, tensor in outputs.items():
        _validate_output_tensor(name, tensor, expected_shapes[name])
    return outputs


def _extract_mask_logits(raw_output: object, output_spec: dict[str, object] | None = None) -> tuple[torch.Tensor, list[str]]:
    resolved_output_spec = output_spec or OUTPUT_SPEC
    outputs = _extract_expected_outputs(raw_output, resolved_output_spec)
    primary_name = str(resolved_output_spec.get("form", "mask_logits"))
    return outputs[primary_name], list(outputs.keys())


def _validate_output_tensor(name: str, tensor: torch.Tensor, expected_shape: list[int]) -> None:
    expected_tail = tuple(expected_shape)
    if tensor.ndim != len(expected_tail) + 1 or tuple(tensor.shape[1:]) != expected_tail:
        expected = ", ".join(str(dim) for dim in expected_tail)
        raise SmokeTestError(f"bad output shape for '{name}': {list(tensor.shape)}; expected [B, {expected}]")
    if not torch.is_floating_point(tensor):
        raise SmokeTestError(f"bad output dtype for '{name}': {tensor.dtype}; expected floating point logits")


def _spec_shape(spec: dict[str, object], label: str) -> list[int]:
    shape = spec.get("shape")
    if not isinstance(shape, list) or not shape or not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise SmokeTestError(f"{label} must contain a positive integer shape list")
    return list(shape)


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
