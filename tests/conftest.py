from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from typer.testing import CliRunner


@dataclass
class CliCompleted:
    returncode: int
    stdout: str
    stderr: str


@contextmanager
def _pushd(path: Path | None):
    if path is None:
        yield
        return
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def invoke_typer_cli(app, args: Iterable[str], cwd: Path | None = None) -> CliCompleted:
    """Run a Typer app in-process with a subprocess.CompletedProcess-like result.

    Most CLI tests only need command-line parsing, stdout/stderr, and exit status.
    Avoiding a fresh Python interpreter for every assertion keeps the suite usable
    in slow VMs while preserving CLI-level coverage.
    """

    with _pushd(cwd):
        result = CliRunner().invoke(app, list(args))
    stderr = result.stderr
    if not stderr and result.exception is not None:
        cause = getattr(result.exception, "__cause__", None)
        stderr = str(cause or result.exception)
    return CliCompleted(returncode=result.exit_code, stdout=result.stdout, stderr=stderr)

# External Research Problem package roots used by disentangling regression tests.
GVCCS_RESEARCH_PROBLEM_ROOT = Path('/home/iross/code/gvccs-research-problem')
if GVCCS_RESEARCH_PROBLEM_ROOT.exists():
    import sys
    sys.path.insert(0, str(GVCCS_RESEARCH_PROBLEM_ROOT))
