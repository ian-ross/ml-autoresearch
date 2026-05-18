import subprocess
import sys
from pathlib import Path


def test_cli_module_import_does_not_import_torch() -> None:
    code = """
import builtins
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError(f'torch imported during CLI import: {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import ml_autoresearch.cli
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr


def test_agent_cli_help_does_not_import_torch() -> None:
    code = """
import builtins
import sys
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError(f'torch imported during agent CLI help: {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = ['ml-autoresearch-agent', '--help']
from ml_autoresearch.agent_cli import main
main()
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr


def test_agent_safe_validate_candidate_command_does_not_import_torch(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: static_candidate
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
    (candidate / "model.py").write_text("import torch\nraise RuntimeError('model.py imported')\n")
    code = f"""
import builtins
import sys
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError(f'torch imported during agent-safe validation: {{name}}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = [
    'ml-autoresearch-agent',
    'validate-candidate',
    '--candidate',
    {str(candidate)!r},
    '--no-require-proposal',
]
from ml_autoresearch.agent_cli import main
main()
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
