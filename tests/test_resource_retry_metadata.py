import json
from pathlib import Path

import yaml

from ml_autoresearch.execution import OperationResult
from ml_autoresearch.runs import RunStatus, is_resource_failure
from research_problem_helpers import run_candidate_with_synthetic_fixture


VALID_MODEL = """
from torch import nn
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
    def forward(self, x):
        return {'mask_logits': self.conv(x)}
def build_model(input_spec, output_spec):
    return Tiny()
""".strip() + "\n"


def write_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: synthetic_candidate
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
    (candidate / "model.py").write_text(VALID_MODEL)
    return candidate


class SimulatedResourceBackend:
    name = "simulated_resource"

    def __init__(self, *, fail_when_batch_size_at_least: int | None = None):
        self.fail_when_batch_size_at_least = fail_when_batch_size_at_least
        self.batch_sizes: list[int] = []

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        path = Path(run_dir)
        (path / "outputs" / "model_summary.json").write_text("{}\n")
        (path / "outputs" / "logs" / "smoke_test.log").write_text("ok\n")
        return OperationResult(backend=self.name, operation="smoke_test")

    def train_research_problem(
        self,
        run_dir: str | Path,
        provider_config,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        path = Path(run_dir)
        manifest = yaml.safe_load((path / "resolved_manifest.yaml").read_text())
        batch_size = int(manifest["training"]["batch_size"])
        self.batch_sizes.append(batch_size)
        if self.fail_when_batch_size_at_least is not None and batch_size >= self.fail_when_batch_size_at_least:
            raise RuntimeError(f"CUDA out of memory while allocating test tensor at batch_size={batch_size}")
        outputs = path / "outputs"
        (outputs / "logs" / "training.log").write_text("simulated success\n")
        (outputs / "metrics.jsonl").write_text('{"split":"val","val/dice":0.5}\n')
        final = {"val/dice": 0.5, "artifacts": {"best_metrics": "outputs/best_metrics.json"}}
        (outputs / "final_metrics.json").write_text(json.dumps(final) + "\n")
        (outputs / "best_metrics.json").write_text(json.dumps({"selection_metric": "val/dice", "selection_value": 0.5}) + "\n")
        return OperationResult(backend=self.name, operation="train_research_problem")


def test_resource_failure_classifier_recognizes_gpu_oom():
    assert is_resource_failure("CUDA out of memory. Tried to allocate 10 GiB")
    assert is_resource_failure("CUBLAS_STATUS_ALLOC_FAILED")
    assert not is_resource_failure("shape mismatch in candidate output")


def test_no_retry_records_requested_and_effective_batch_size(tmp_path: Path):
    candidate = write_candidate(tmp_path)
    backend = SimulatedResourceBackend()

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", backend=backend)

    assert run.status == RunStatus.COMPLETED
    assert backend.batch_sizes == [2]
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    retry = metadata["training_lifecycle"]["resource_retry"]
    assert retry["requested_batch_size"] == 2
    assert retry["effective_batch_size"] == 2
    assert retry["retry_count"] == 0
    assert retry["exhausted"] is False
    assert retry["attempts"] == [{"attempt": 1, "batch_size": 2, "outcome": "completed"}]


def test_resource_retry_success_lowers_batch_size_and_records_attempts(tmp_path: Path):
    candidate = write_candidate(tmp_path)
    backend = SimulatedResourceBackend(fail_when_batch_size_at_least=2)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", backend=backend)

    assert run.status == RunStatus.COMPLETED
    assert backend.batch_sizes == [2, 1]
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    retry = metadata["training_lifecycle"]["resource_retry"]
    assert metadata["training_lifecycle"]["status"] == "completed_after_resource_retry"
    assert retry["requested_batch_size"] == 2
    assert retry["effective_batch_size"] == 1
    assert retry["retry_count"] == 1
    assert retry["attempts"][0]["failure_classification"] == "resource_failure"
    assert retry["attempts"][0]["next_batch_size"] == 1
    resolved = yaml.safe_load((run.run_dir / "resolved_manifest.yaml").read_text())
    assert resolved["training"]["batch_size_requested"] == 2
    assert resolved["training"]["batch_size_effective"] == 1
    assert "retry succeeded" in (run.run_dir / "outputs" / "logs" / "resource_retry.log").read_text()


def test_resource_retry_exhaustion_fails_as_resource_failure_and_is_bounded(tmp_path: Path):
    candidate = write_candidate(tmp_path)
    backend = SimulatedResourceBackend(fail_when_batch_size_at_least=1)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", backend=backend)

    assert run.status == RunStatus.FAILED
    assert run.failure_classification == "resource_failure"
    assert backend.batch_sizes == [2, 1]
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["failure_classification"] == "resource_failure"
    assert "Resource Failure retry exhausted" in metadata["training_failure_reason"]
    retry = metadata["training_lifecycle"]["resource_retry"]
    assert metadata["training_lifecycle"]["status"] == "resource_retry_exhausted"
    assert retry["exhausted"] is True
    assert retry["effective_batch_size"] == 1
    assert len(retry["attempts"]) == 2
    assert "retry exhausted" in (run.run_dir / "outputs" / "logs" / "resource_retry.log").read_text()
