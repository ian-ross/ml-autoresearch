import subprocess
import sys


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
