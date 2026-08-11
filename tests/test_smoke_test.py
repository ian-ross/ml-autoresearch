import json
from pathlib import Path

import pytest

from ml_autoresearch.candidate_execution_config import load_configured_research_problem_registry
from ml_autoresearch.execution import NativeBackend
from ml_autoresearch.runs import (
    RunStatus,
    prepare_candidate_submission,
    smoke_test_prepared_run,
    submit_candidate,
)
from research_problem_helpers import write_fake_candidate_execution_config, write_fake_research_problem_package


def write_candidate(root: Path, model_py: str, *, auxiliary_targets: str = "") -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: smoke_candidate
input_mode: single_frame_rgb
output_form: mask_logits
{auxiliary_targets}training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".format(auxiliary_targets=auxiliary_targets).strip()
        + "\n"
    )
    (candidate / "model.py").write_text(model_py)
    return candidate


VALID_MODEL = """
import torch
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
    def forward(self, x):
        return {"mask_logits": self.conv(x)}

def build_model(input_spec, output_spec):
    assert input_spec == {"mode": "single_frame_rgb", "shape": [3, 128, 128]}
    assert output_spec == {"form": "mask_logits", "shape": [1, 128, 128]}
    return Tiny()
""".strip() + "\n"


def test_trusted_provider_bootstrap_failure_during_smoke_is_harness_owned(tmp_path: Path) -> None:
    candidate = write_candidate(tmp_path, VALID_MODEL)
    write_fake_research_problem_package(tmp_path)
    write_fake_candidate_execution_config(tmp_path)
    registry = load_configured_research_problem_registry(tmp_path)
    assert registry is not None
    prepared = prepare_candidate_submission(
        candidate,
        tmp_path / "runs",
        backend=NativeBackend(),
        research_problem_registry=registry,
    )
    manifest_path = prepared.run_dir / "resolved_manifest.yaml"
    manifest_path.write_text(
        manifest_path.read_text().replace(
            "fake_problem.research_problem:build_spec",
            "missing_trusted_provider.research_problem:build_spec",
        )
    )

    run = smoke_test_prepared_run(prepared.run_dir, backend=NativeBackend())

    assert run.status == RunStatus.SMOKE_FAILED
    assert run.failure_classification == "harness_failure"
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["failure_classification"] == "harness_failure"
    assert "missing_trusted_provider" in metadata["smoke_failure_reason"]


def test_submit_candidate_smoke_tests_model_and_writes_artifacts(tmp_path: Path):
    candidate = write_candidate(tmp_path, VALID_MODEL)

    run = submit_candidate(candidate, tmp_path / "runs")

    assert run.status == RunStatus.ACCEPTED
    run_dir = run.run_dir
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "accepted"
    assert metadata["smoke_failure_reason"] is None

    summary = json.loads((run_dir / "outputs" / "model_summary.json").read_text())
    assert summary["parameter_count"] == 4
    assert summary["input_spec"] == {"mode": "single_frame_rgb", "shape": [3, 128, 128]}
    assert summary["output_spec"] == {"form": "mask_logits", "shape": [1, 128, 128]}
    assert summary["output"]["names"] == ["mask_logits"]
    assert summary["output"]["shape"] == [2, 1, 128, 128]
    assert "Smoke test accepted" in (run_dir / "outputs" / "logs" / "smoke_test.log").read_text()


@pytest.mark.parametrize(
    ("model_py", "expected_reason"),
    [
        ("x = 1\n", "missing required build_model"),
        (
            "def build_model(input_spec, output_spec):\n    raise RuntimeError('boom')\n",
            "build_model failed: boom",
        ),
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def forward(self, x):\n        return {'line_logits': x[:, :1]}\ndef build_model(i, o):\n    return Bad()\n",
            "unexpected output keys",
        ),
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def forward(self, x):\n        return x[:, :1, :64, :64]\ndef build_model(i, o):\n    return Bad()\n",
            "bad output shape",
        ),
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def forward(self, x):\n        return (x[:, :1] > 0)\ndef build_model(i, o):\n    return Bad()\n",
            "bad output dtype",
        ),
    ],
)
def test_smoke_failures_are_recorded_in_metadata_and_log(tmp_path: Path, model_py: str, expected_reason: str):
    candidate = write_candidate(tmp_path, model_py)

    run = submit_candidate(candidate, tmp_path / "runs")

    assert run.status == RunStatus.SMOKE_FAILED
    assert run.failure_classification == "candidate_bug"
    assert run.rejection_reason and expected_reason in run.rejection_reason
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "smoke_failed"
    assert expected_reason in metadata["smoke_failure_reason"]
    assert expected_reason in (run.run_dir / "outputs" / "logs" / "smoke_test.log").read_text()


@pytest.mark.parametrize(
    ("model_py", "checkpoint", "quantity"),
    [
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def __init__(self):\n        super().__init__(); self.scale = nn.Parameter(torch.tensor(1.0))\n    def forward(self, x):\n        return {'mask_logits': x[:, :1] + self.scale * torch.tensor(float('nan'))}\ndef build_model(i, o): return Bad()\n",
            "forward_outputs",
            "output.mask_logits",
        ),
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def __init__(self):\n        super().__init__(); self.scale = nn.Parameter(torch.tensor(1.0))\n    def forward(self, x):\n        return {'mask_logits': torch.full_like(x[:, :1], 3e38) * self.scale}\ndef build_model(i, o): return Bad()\n",
            "loss",
            "loss.synthetic",
        ),
        (
            "import torch\nfrom torch import nn\nclass BadGradient(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, value, x): return value * x[:, :1]\n    @staticmethod\n    def backward(ctx, gradient): return torch.tensor(float('inf')), None\nclass Bad(nn.Module):\n    def __init__(self):\n        super().__init__(); self.scale = nn.Parameter(torch.tensor(1.0))\n    def forward(self, x): return {'mask_logits': BadGradient.apply(self.scale, x)}\ndef build_model(i, o): return Bad()\n",
            "gradients",
            "gradient.scale",
        ),
    ],
)
def test_smoke_fails_closed_on_nonfinite_output_loss_or_gradient(
    tmp_path: Path, model_py: str, checkpoint: str, quantity: str
) -> None:
    run = submit_candidate(write_candidate(tmp_path, model_py), tmp_path / "runs")

    assert run.status == RunStatus.SMOKE_FAILED
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "smoke"
    assert diagnostic["checkpoint"] == checkpoint
    assert diagnostic["failing_quantity"] == quantity


def test_trusted_parameter_budget_is_enforced_and_recorded(tmp_path: Path):
    candidate = write_candidate(tmp_path, VALID_MODEL)

    rejected = submit_candidate(candidate, tmp_path / "rejected-runs", max_parameters=3)
    accepted = submit_candidate(candidate, tmp_path / "accepted-runs", max_parameters=4)

    assert rejected.status == RunStatus.SMOKE_FAILED
    assert rejected.rejection_reason and "parameter-budget violation" in rejected.rejection_reason
    assert accepted.status == RunStatus.ACCEPTED
    summary = json.loads((accepted.run_dir / "outputs" / "model_summary.json").read_text())
    assert summary["parameter_count"] == 4
    assert summary["parameter_budget"] == {"max_parameters": 4}


LINE_AUX_MANIFEST = """auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: 0.25
"""


LINE_AUX_MODEL = """
import torch
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = nn.Conv2d(3, 1, kernel_size=1)
        self.line = nn.Conv2d(3, 1, kernel_size=1)
    def forward(self, x):
        return {"mask_logits": self.mask(x), "line_logits": self.line(x)}

def build_model(input_spec, output_spec):
    assert output_spec == {
        "form": "mask_logits",
        "shape": [1, 128, 128],
        "auxiliary_outputs": [{"target": "line", "name": "line_logits", "shape": [1, 128, 128]}],
    }
    return Tiny()
""".strip() + "\n"


def test_smoke_test_requires_requested_line_logits(tmp_path: Path):
    candidate = write_candidate(tmp_path, LINE_AUX_MODEL, auxiliary_targets=LINE_AUX_MANIFEST)

    run = submit_candidate(candidate, tmp_path / "runs")

    assert run.status == RunStatus.ACCEPTED
    summary = json.loads((run.run_dir / "outputs" / "model_summary.json").read_text())
    assert summary["output_spec"]["auxiliary_outputs"] == [
        {"target": "line", "name": "line_logits", "shape": [1, 128, 128]}
    ]
    assert summary["output"]["names"] == ["mask_logits", "line_logits"]


BOUNDARY_AUX_MANIFEST = """auxiliary_targets:
  - name: boundary
    output: boundary_logits
    loss: weighted_bce
    weight: 0.10
"""


BOUNDARY_AUX_MODEL = """
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = nn.Conv2d(3, 1, kernel_size=1)
        self.boundary = nn.Conv2d(3, 1, kernel_size=1)
    def forward(self, x):
        return {"mask_logits": self.mask(x), "boundary_logits": self.boundary(x)}

def build_model(input_spec, output_spec):
    assert output_spec == {
        "form": "mask_logits",
        "shape": [1, 128, 128],
        "auxiliary_outputs": [{"target": "boundary", "name": "boundary_logits", "shape": [1, 128, 128]}],
    }
    return Tiny()
""".strip() + "\n"


def test_smoke_test_requires_requested_boundary_logits(tmp_path: Path):
    candidate = write_candidate(tmp_path, BOUNDARY_AUX_MODEL, auxiliary_targets=BOUNDARY_AUX_MANIFEST)

    run = submit_candidate(candidate, tmp_path / "runs")

    assert run.status == RunStatus.ACCEPTED
    summary = json.loads((run.run_dir / "outputs" / "model_summary.json").read_text())
    assert summary["output_spec"]["auxiliary_outputs"] == [
        {"target": "boundary", "name": "boundary_logits", "shape": [1, 128, 128]}
    ]
    assert summary["output"]["names"] == ["mask_logits", "boundary_logits"]


@pytest.mark.parametrize(
    ("model_py", "expected_reason"),
    [
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def forward(self, x):\n        return {'mask_logits': x[:, :1]}\ndef build_model(i, o):\n    return Bad()\n",
            "unexpected output keys",
        ),
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def forward(self, x):\n        return {'mask_logits': x[:, :1], 'line_logits': x[:, :1], 'extra_logits': x[:, :1]}\ndef build_model(i, o):\n    return Bad()\n",
            "unexpected output keys",
        ),
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def forward(self, x):\n        return x[:, :1]\ndef build_model(i, o):\n    return Bad()\n",
            "tensor output shorthand is only valid for mask-only candidates",
        ),
        (
            "import torch\nfrom torch import nn\nclass Bad(nn.Module):\n    def forward(self, x):\n        return {'mask_logits': x[:, :1], 'line_logits': x[:, :1, :64, :64]}\ndef build_model(i, o):\n    return Bad()\n",
            "bad output shape for 'line_logits'",
        ),
    ],
)
def test_auxiliary_output_smoke_failures_are_recorded(tmp_path: Path, model_py: str, expected_reason: str):
    candidate = write_candidate(tmp_path, model_py, auxiliary_targets=LINE_AUX_MANIFEST)

    run = submit_candidate(candidate, tmp_path / "runs")

    assert run.status == RunStatus.SMOKE_FAILED
    assert run.rejection_reason and expected_reason in run.rejection_reason


def test_smoke_specs_from_resolved_manifest_use_fake_research_problem_spec(tmp_path: Path):
    from ml_autoresearch.research_problems import ResearchProblemSpec, ResearchProblemSpecRegistry
    from ml_autoresearch.smoke import smoke_specs_from_resolved_manifest

    fake = ResearchProblemSpec(
        id="fake_temporal_problem",
        version="test-v0",
        input_modes=("tiny_clip_rgb",),
        input_specs={"tiny_clip_rgb": {"mode": "tiny_clip_rgb", "shape": [5, 3, 16, 16]}},
        output_forms=("tiny_mask_logits",),
        output_specs={"tiny_mask_logits": {"form": "tiny_mask_logits", "shape": [1, 16, 16]}},
        auxiliary_targets=("edge",),
        auxiliary_outputs={"edge": "edge_logits"},
        auxiliary_output_shapes={"edge": [1, 16, 16]},
        losses=("tiny_loss",),
        auxiliary_losses=("tiny_aux_loss",),
        optimizers=("sgd",),
        sampling_policies=("sequential",),
        augmentation_policies=("none",),
        primary_metric="val/tiny_score",
    )
    registry = ResearchProblemSpecRegistry((fake,), default_id=fake.id)
    resolved_manifest = tmp_path / "resolved_manifest.yaml"
    resolved_manifest.write_text(
        """
research_problem:
  id: fake_temporal_problem
  version: test-v0
input_mode: tiny_clip_rgb
output_form: tiny_mask_logits
auxiliary_targets:
  - name: edge
    output: edge_logits
""".strip() + "\n"
    )

    input_spec, output_spec = smoke_specs_from_resolved_manifest(resolved_manifest, research_problem_registry=registry)

    assert input_spec == {"mode": "tiny_clip_rgb", "shape": [5, 3, 16, 16]}
    assert output_spec == {
        "form": "tiny_mask_logits",
        "shape": [1, 16, 16],
        "auxiliary_outputs": [{"target": "edge", "name": "edge_logits", "shape": [1, 16, 16]}],
    }


def test_smoke_specs_from_resolved_manifest_use_container_research_problem_root_override(
    tmp_path: Path, monkeypatch
):
    from ml_autoresearch.smoke import smoke_specs_from_resolved_manifest

    package_root = tmp_path / "research-problem"
    package = package_root / "tiny_problem"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='tiny_problem', version='test-v0', contract_version='v0',\n"
        "        input_modes=('tiny_rgb',), input_specs={'tiny_rgb': {'mode': 'tiny_rgb', 'shape': [3, 8, 8]}},\n"
        "        output_forms=('tiny_mask_logits',), output_specs={'tiny_mask_logits': {'form': 'tiny_mask_logits', 'shape': [1, 8, 8]}},\n"
        "        losses=('tiny_loss',), optimizers=('sgd',), sampling_policies=('sequential',), augmentation_policies=('none',),\n"
        "        primary_metric='val/tiny_score',\n"
        "    )\n"
    )
    resolved_manifest = tmp_path / "resolved_manifest.yaml"
    resolved_manifest.write_text(
        """
research_problem:
  id: tiny_problem
  version: test-v0
  contract_version: v0
  provider:
    target: tiny_problem.research_problem:build_spec
    resolved_package_root: /host/path/that/does/not/exist
input_mode: tiny_rgb
output_form: tiny_mask_logits
""".lstrip()
    )
    monkeypatch.setenv("ML_AUTORESEARCH_RESEARCH_PROBLEM_ROOT", str(package_root))

    input_spec, output_spec = smoke_specs_from_resolved_manifest(resolved_manifest)

    assert input_spec == {"mode": "tiny_rgb", "shape": [3, 8, 8]}
    assert output_spec == {"form": "tiny_mask_logits", "shape": [1, 8, 8]}
