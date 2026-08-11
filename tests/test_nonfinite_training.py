from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml_autoresearch.execution import OperationResult
from ml_autoresearch.research_problems import ResearchProblemProviderConfig
from ml_autoresearch.runs import RunFailureClassification, RunStatus, run_candidate_with_research_problem
from research_problem_helpers import write_fake_research_problem_package


def _write_candidate_with_nonfinite_initial_parameter(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: nonfinite_smoke_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        """
import torch
from torch import nn

class NonFiniteInitialParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float("nan")))

    def forward(self, inputs):
        return inputs[:, :1] * self.scale

def build_model(input_spec, output_spec):
    return NonFiniteInitialParameter()
""".strip()
        + "\n"
    )
    return candidate


def _write_candidate_that_becomes_nonfinite_during_training(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: nonfinite_training_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 2
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        """
import torch
from torch import nn

class NonFiniteDuringTraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return {"mask_logits": torch.exp(inputs[:, :1] * self.scale * 1000.0)}

def build_model(input_spec, output_spec):
    return NonFiniteDuringTraining()
""".strip()
        + "\n"
    )
    return candidate


def _write_finite_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: finite_training_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        """
from torch import nn

class FiniteCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 1, 1)

    def forward(self, inputs):
        return {"mask_logits": self.layer(inputs)}

def build_model(input_spec, output_spec):
    return FiniteCandidate()
""".strip()
        + "\n"
    )
    return candidate


def _provider(tmp_path: Path) -> ResearchProblemProviderConfig:
    return ResearchProblemProviderConfig(
        id="fake_problem",
        package_root=tmp_path,
        provider_target="fake_problem.research_problem:build_spec",
        expected_contract_version="v0",
        data_config={"sample_count": 4},
    )


def test_nonfinite_initial_parameter_fails_smoke_with_diagnostic(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    run = run_candidate_with_research_problem(
        _write_candidate_with_nonfinite_initial_parameter(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.SMOKE_FAILED
    assert run.failure_classification == RunFailureClassification.CANDIDATE_BUG
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "smoke"
    assert diagnostic["checkpoint"] == "initial_parameters"
    assert diagnostic["failing_quantity"] == "parameter.scale"
    assert diagnostic["epoch"] is None
    assert diagnostic["batch"] == 0


def test_nonfinite_training_fails_first_batch_with_bounded_diagnostic_and_no_resource_retry(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    provider = _provider(tmp_path)
    ledger = tmp_path / "research-ledger.jsonl"

    run = run_candidate_with_research_problem(
        _write_candidate_that_becomes_nonfinite_during_training(tmp_path),
        tmp_path / "runs",
        provider,
        max_samples=4,
        max_prediction_samples=1,
        ledger_path=ledger,
    )

    assert run.status == RunStatus.FAILED
    assert run.failure_classification == RunFailureClassification.CANDIDATE_BUG
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["training_failure_reason"].startswith("non_finite_training_state:")
    assert metadata["failure_classification"] == "candidate_bug"

    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["schema_version"] == 1
    assert diagnostic["failure_type"] == "non_finite_training_state"
    assert diagnostic["phase"] == "train"
    assert diagnostic["checkpoint"] == "forward_outputs"
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] == 0
    assert diagnostic["failing_quantity"] == "output.mask_logits"
    assert diagnostic["counts"]["total"] > 0
    assert diagnostic["counts"]["finite"] + diagnostic["counts"]["nonfinite"] == diagnostic["counts"]["total"]
    assert diagnostic["counts"]["nonfinite"] > 0
    assert set(diagnostic["counts"]) == {"total", "finite", "nonfinite", "nan", "positive_infinity", "negative_infinity"}
    assert "values" not in json.dumps(diagnostic)
    assert (run.run_dir / "outputs" / "nonfinite_diagnostic.json").stat().st_size < 64 * 1024

    metric_rows = (run.run_dir / "outputs" / "metrics.jsonl").read_text().splitlines()
    assert metric_rows == []
    assert not (run.run_dir / "outputs" / "logs" / "resource_retry.log").exists()
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [event["event_type"] for event in events][-1] == "run_failed"
    assert events[-1]["failure_classification"] == "candidate_bug"


def _write_candidate_with_nonfinite_training_gradient(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: nonfinite_gradient_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        """
import torch
from torch import nn

class InputDependentGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, inputs):
        ctx.save_for_backward(inputs)
        return value * inputs[:, :1]

    @staticmethod
    def backward(ctx, gradient):
        (inputs,) = ctx.saved_tensors
        multiplier = torch.where(inputs[:, :1] == 0, torch.ones_like(inputs[:, :1]), torch.full_like(inputs[:, :1], float("inf")))
        return (gradient * inputs[:, :1] * multiplier).sum(), None

class NonFiniteGradientCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs):
        return {"mask_logits": InputDependentGradient.apply(self.scale, inputs)}

def build_model(input_spec, output_spec):
    return NonFiniteGradientCandidate()
""".strip()
        + "\n"
    )
    return candidate


def test_nonfinite_primary_loss_fails_before_backward(tmp_path: Path) -> None:
    package = write_fake_research_problem_package(tmp_path)
    provider_source = package / "research_problem.py"
    provider_source.write_text(
        provider_source.read_text().replace(
            "        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target_mask)\n",
            "        return logits.sum() * torch.tensor(float('nan'))\n",
        )
    )

    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["checkpoint"] == "losses"
    assert diagnostic["failing_quantity"] == "loss.primary"
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] == 0


def test_nonfinite_gradient_fails_before_optimizer_step(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)

    run = run_candidate_with_research_problem(
        _write_candidate_with_nonfinite_training_gradient(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["checkpoint"] == "gradients"
    assert diagnostic["failing_quantity"] == "gradient.scale"
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] == 0
    assert not (run.run_dir / "outputs" / "models" / "best_epoch_model.pt").exists()


def _write_candidate_with_nonfinite_validation_output(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: nonfinite_validation_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        """
import torch
from torch import nn

class NonFiniteValidationCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 1, 1)

    def forward(self, inputs):
        if self.training:
            return {"mask_logits": self.layer(inputs)}
        return {"mask_logits": torch.exp(inputs[:, :1] * 1000.0)}

def build_model(input_spec, output_spec):
    return NonFiniteValidationCandidate()
""".strip()
        + "\n"
    )
    return candidate


def test_nonfinite_parameter_fails_immediately_after_optimizer_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_fake_research_problem_package(tmp_path)
    original_step = torch.optim.AdamW.step

    def corrupt_after_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        optimizer.param_groups[0]["params"][0].data.fill_(float("inf"))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_after_step)
    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["checkpoint"] == "parameters_after_step"
    assert diagnostic["failing_quantity"].startswith("parameter.")
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] == 0


def test_nonfinite_validation_output_fails_before_aggregate_metrics(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    run = run_candidate_with_research_problem(
        _write_candidate_with_nonfinite_validation_output(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "validation"
    assert diagnostic["checkpoint"] == "forward_outputs"
    assert diagnostic["failing_quantity"] == "output.mask_logits"
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] == 0


def test_nonfinite_aggregate_epoch_value_fails_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_fake_research_problem_package(tmp_path)
    monkeypatch.setattr(
        "ml_autoresearch.training._evaluate",
        lambda *args, **kwargs: (
            {"val/dice": 0.5, "val/loss": float("nan")},
            {"postprocessing": {"backend": "torch_cpu", "timings_seconds": {"artifact_filter": 0.01}}},
        ),
    )

    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    assert run.failure_classification == RunFailureClassification.HARNESS_FAILURE
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "validation"
    assert diagnostic["checkpoint"] == "aggregate_epoch_values"
    assert diagnostic["failing_quantity"] == "metric.val/loss"
    assert diagnostic["epoch"] == 1
    assert not (run.run_dir / "outputs" / "models" / "best_epoch_model.pt").exists()
    assert not (run.run_dir / "outputs" / "validation_postprocessing" / "epoch_001.json").exists()


def test_nonfinite_selection_metric_fails_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_fake_research_problem_package(tmp_path)
    monkeypatch.setattr(
        "ml_autoresearch.training._evaluate",
        lambda *args, **kwargs: ({"val/dice": float("nan"), "val/loss": 0.25}, None),
    )

    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    assert run.failure_classification == RunFailureClassification.HARNESS_FAILURE
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "validation"
    assert diagnostic["checkpoint"] == "selection_metric"
    assert diagnostic["failing_quantity"] == "metric.val/dice"
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] is None
    assert not (run.run_dir / "outputs" / "models" / "best_epoch_model.pt").exists()


def test_nonfinite_validation_loss_fails_before_aggregation(tmp_path: Path) -> None:
    package = write_fake_research_problem_package(tmp_path)
    provider_source = package / "research_problem.py"
    provider_source.write_text(
        provider_source.read_text().replace(
            "        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target_mask)\n",
            "        if not torch.is_grad_enabled():\n            return logits.sum() * torch.tensor(float('nan'))\n        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target_mask)\n",
        )
    )

    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "validation"
    assert diagnostic["checkpoint"] == "losses"
    assert diagnostic["failing_quantity"] == "loss.primary"
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] == 0


class NonFiniteArtifactBackend:
    name = "nonfinite-artifact"

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        outputs = Path(run_dir) / "outputs"
        (outputs / "logs").mkdir(parents=True, exist_ok=True)
        (outputs / "logs" / "smoke_test.log").write_text("accepted\n")
        (outputs / "model_summary.json").write_text("{}\n")
        return OperationResult(backend=self.name, operation="smoke_test")

    def train_research_problem(self, run_dir, provider_config, **kwargs) -> OperationResult:
        outputs = Path(run_dir) / "outputs"
        (outputs / "logs").mkdir(parents=True, exist_ok=True)
        (outputs / "models").mkdir(parents=True, exist_ok=True)
        (outputs / "logs" / "training.log").write_text("completed with invalid artifacts\n")
        (outputs / "metrics.jsonl").write_text('{"split":"val","val/dice":NaN}\n')
        (outputs / "final_metrics.json").write_text(
            '{"val/dice":NaN,"artifacts":{"best_metrics":"outputs/best_metrics.json","best_epoch_model":"outputs/models/best_epoch_model.pt"}}\n'
        )
        (outputs / "best_metrics.json").write_text(
            '{"selection_metric":"val/dice","selection_value":NaN,"model_artifact":"outputs/models/best_epoch_model.pt"}\n'
        )
        torch.save(
            {"model_state_dict": {"layer.weight": torch.tensor([float("nan")])}},
            outputs / "models" / "best_epoch_model.pt",
        )
        return OperationResult(backend=self.name, operation="train_research_problem")


class WrappedHarnessNonFiniteBackend(NonFiniteArtifactBackend):
    def train_research_problem(self, run_dir, provider_config, **kwargs) -> OperationResult:
        outputs = Path(run_dir) / "outputs"
        (outputs / "nonfinite_diagnostic.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "failure_type": "non_finite_training_state",
                    "phase": "validation",
                    "checkpoint": "aggregate_metrics",
                    "epoch": 1,
                    "batch": None,
                    "failing_quantity": "metric.val/dice",
                    "failure_classification": "harness_failure",
                    "counts": {
                        "total": 1,
                        "finite": 0,
                        "nonfinite": 1,
                        "nan": 1,
                        "positive_infinity": 0,
                        "negative_infinity": 0,
                    },
                }
            )
            + "\n"
        )
        raise RuntimeError("Docker Research Problem training failed: non_finite_training_state: metric.val/dice")


class NonFiniteCheckpointBackend(NonFiniteArtifactBackend):
    def train_research_problem(self, run_dir, provider_config, **kwargs) -> OperationResult:
        outputs = Path(run_dir) / "outputs"
        (outputs / "logs").mkdir(parents=True, exist_ok=True)
        (outputs / "models").mkdir(parents=True, exist_ok=True)
        (outputs / "logs" / "training.log").write_text("completed with invalid checkpoint\n")
        (outputs / "metrics.jsonl").write_text('{"split":"val","val/dice":0.5}\n')
        (outputs / "final_metrics.json").write_text(
            '{"val/dice":0.5,"artifacts":{"best_metrics":"outputs/best_metrics.json","best_epoch_model":"outputs/models/best_epoch_model.pt"}}\n'
        )
        (outputs / "best_metrics.json").write_text(
            '{"selection_metric":"val/dice","selection_value":0.5,"model_artifact":"outputs/models/best_epoch_model.pt"}\n'
        )
        torch.save(
            {"model_state_dict": {"layer.weight": torch.tensor([float("inf")])}},
            outputs / "models" / "best_epoch_model.pt",
        )
        return OperationResult(backend=self.name, operation="train_research_problem")


def test_wrapped_backend_preserves_harness_nonfinite_classification(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        backend=WrappedHarnessNonFiniteBackend(),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    assert run.failure_classification == RunFailureClassification.HARNESS_FAILURE
    assert not (run.run_dir / "outputs" / "logs" / "resource_retry.log").exists()


def test_terminal_artifact_validation_rejects_nonfinite_metrics_and_checkpoint(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    ledger = tmp_path / "research-ledger.jsonl"
    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        backend=NonFiniteArtifactBackend(),
        ledger_path=ledger,
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    assert run.failure_classification == RunFailureClassification.CANDIDATE_BUG
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "terminal_validation"
    assert diagnostic["checkpoint"] == "metrics_artifacts"
    assert diagnostic["failing_quantity"] == "metric.metrics.jsonl[0].val/dice"
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert events[-1]["event_type"] == "run_failed"
    assert all(event["event_type"] != "run_completed" for event in events)


def test_terminal_artifact_validation_rejects_nonfinite_checkpoint_tensor(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        backend=NonFiniteCheckpointBackend(),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "terminal_validation"
    assert diagnostic["checkpoint"] == "checkpoint_tensors"
    assert diagnostic["failing_quantity"] == "checkpoint.model_state_dict.layer.weight"


def test_nonfinite_trusted_aggregate_metric_fails_as_harness_failure(tmp_path: Path) -> None:
    package = write_fake_research_problem_package(tmp_path)
    provider_source = package / "research_problem.py"
    provider_source.write_text(
        provider_source.read_text().replace(
            "        return {\"val/dice\": float(result['dice'])}\n",
            "        return {\"val/dice\": float('nan')}\n",
        )
    )

    run = run_candidate_with_research_problem(
        _write_finite_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
    )

    assert run.status == RunStatus.FAILED
    assert run.failure_classification == RunFailureClassification.HARNESS_FAILURE
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "validation"
    assert diagnostic["checkpoint"] == "aggregate_metrics"
    assert diagnostic["failing_quantity"] == "metric.val/dice"
    assert diagnostic["failure_classification"] == "harness_failure"
    assert diagnostic["epoch"] == 1
    assert diagnostic["batch"] is None
