import subprocess
import sys
from pathlib import Path

from research_problem_helpers import write_static_candidate_execution_config


def test_cli_and_agent_safe_commands_do_not_import_torch(tmp_path: Path) -> None:
    write_static_candidate_execution_config(tmp_path)
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
        raise RuntimeError(f'torch imported during lazy-import check: {{name}}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import ml_autoresearch.cli

sys.argv = ['ml-autoresearch-agent', '--help']
from ml_autoresearch.agent_cli import main
try:
    main()
except SystemExit as exc:
    if exc.code not in (0, None):
        raise

sys.argv = [
    'ml-autoresearch-agent',
    'validate-candidate',
    '--candidate',
    {str(candidate)!r},
    '--no-require-proposal',
    '--project-root',
    {str(tmp_path)!r},
]
try:
    main()
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
