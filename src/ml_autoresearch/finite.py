"""Trusted finite-state validation for Candidate Execution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from numbers import Real
from pathlib import Path

import torch

from ml_autoresearch.errors import TrainingError

NONFINITE_DIAGNOSTIC = "nonfinite_diagnostic.json"


class NonFiniteStateError(TrainingError):
    """Raised when trusted execution observes non-finite numerical state."""

    def __init__(self, message: str, *, failure_classification: str):
        super().__init__(message)
        self.failure_classification = failure_classification


def require_finite_tensor(
    tensor: torch.Tensor,
    *,
    outputs_dir: str | Path,
    phase: str,
    checkpoint: str,
    failing_quantity: str,
    epoch: int | None = None,
    batch: int | None = None,
    failure_classification: str = "candidate_bug",
) -> None:
    """Fail closed and persist a bounded count-only diagnostic for one tensor."""

    finite_mask = torch.isfinite(tensor.detach())
    if bool(finite_mask.all().item()):
        return
    detached = tensor.detach()
    total = detached.numel()
    finite = int(finite_mask.sum().item())
    bounded_quantity = failing_quantity[:256]
    if detached.is_complex():
        positive_infinity = int(torch.isinf(detached).sum().item())
        negative_infinity = 0
    else:
        positive_infinity = int(torch.isposinf(detached).sum().item())
        negative_infinity = int(torch.isneginf(detached).sum().item())
    diagnostic = {
        "schema_version": 1,
        "failure_type": "non_finite_training_state",
        "phase": phase,
        "checkpoint": checkpoint,
        "epoch": epoch,
        "batch": batch,
        "failing_quantity": bounded_quantity,
        "failure_classification": failure_classification,
        "counts": {
            "total": total,
            "finite": finite,
            "nonfinite": total - finite,
            "nan": int(torch.isnan(detached).sum().item()),
            "positive_infinity": positive_infinity,
            "negative_infinity": negative_infinity,
        },
    }
    write_nonfinite_diagnostic(outputs_dir, diagnostic)
    raise NonFiniteStateError(
        f"non_finite_training_state: {bounded_quantity} contains {total - finite} non-finite values "
        f"at {phase}.{checkpoint} epoch={epoch} batch={batch}",
        failure_classification=failure_classification,
    )


def require_finite_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    outputs_dir: str | Path,
    phase: str,
    checkpoint: str,
    quantity_prefix: str,
    epoch: int | None = None,
    batch: int | None = None,
    failure_classification: str = "candidate_bug",
) -> None:
    """Check a named tensor collection with one host synchronization per device."""

    by_device: dict[torch.device, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    for name, tensor in named_tensors:
        detached = tensor.detach()
        by_device.setdefault(detached.device, []).append(
            (name, tensor, torch.isfinite(detached).all())
        )
    for checks in by_device.values():
        if bool(torch.stack([check for _name, _tensor, check in checks]).all().item()):
            continue
        for name, tensor, check in checks:
            if not bool(check.item()):
                require_finite_tensor(
                    tensor,
                    outputs_dir=outputs_dir,
                    phase=phase,
                    checkpoint=checkpoint,
                    failing_quantity=f"{quantity_prefix}.{name}",
                    epoch=epoch,
                    batch=batch,
                    failure_classification=failure_classification,
                )


def require_finite_json_numbers(
    value: object,
    *,
    outputs_dir: str | Path,
    phase: str,
    checkpoint: str,
    quantity_prefix: str,
    epoch: int | None = None,
    batch: int | None = None,
    failure_classification: str = "candidate_bug",
) -> None:
    """Recursively reject non-finite numeric values in trusted JSON-like artifacts."""

    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_json_numbers(
                item,
                outputs_dir=outputs_dir,
                phase=phase,
                checkpoint=checkpoint,
                quantity_prefix=f"{quantity_prefix}.{key}",
                epoch=epoch,
                batch=batch,
                failure_classification=failure_classification,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite_json_numbers(
                item,
                outputs_dir=outputs_dir,
                phase=phase,
                checkpoint=checkpoint,
                quantity_prefix=f"{quantity_prefix}[{index}]",
                epoch=epoch,
                batch=batch,
                failure_classification=failure_classification,
            )
        return
    if isinstance(value, Real) and not isinstance(value, bool):
        require_finite_tensor(
            torch.as_tensor(value, dtype=torch.float64),
            outputs_dir=outputs_dir,
            phase=phase,
            checkpoint=checkpoint,
            failing_quantity=quantity_prefix,
            epoch=epoch,
            batch=batch,
            failure_classification=failure_classification,
        )


def write_nonfinite_diagnostic(outputs_dir: str | Path, diagnostic: dict[str, object]) -> Path:
    """Write the first bounded diagnostic without replacing earlier failure evidence."""

    path = Path(outputs_dir) / NONFINITE_DIAGNOSTIC
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
