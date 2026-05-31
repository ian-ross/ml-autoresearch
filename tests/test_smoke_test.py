import json
from pathlib import Path

import pytest

from ml_autoresearch.runs import RunStatus, submit_candidate


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
    assert run.rejection_reason and expected_reason in run.rejection_reason
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "smoke_failed"
    assert expected_reason in metadata["smoke_failure_reason"]
    assert expected_reason in (run.run_dir / "outputs" / "logs" / "smoke_test.log").read_text()


def test_parameter_budget_violations_are_smoke_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ml_autoresearch.smoke as smoke

    monkeypatch.setattr(smoke, "MAX_PARAMETER_COUNT", 3)
    candidate = write_candidate(tmp_path, VALID_MODEL)

    run = submit_candidate(candidate, tmp_path / "runs")

    assert run.status == RunStatus.SMOKE_FAILED
    assert run.rejection_reason and "parameter-budget violation" in run.rejection_reason


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
