from pathlib import Path
import tomllib


def test_project_requires_python_312_only() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"


def test_uv_lock_records_python_312_baseline() -> None:
    first_lines = Path("uv.lock").read_text().splitlines()[:5]

    assert 'requires-python = "==3.12.*"' in first_lines


def test_readme_development_setup_names_python_312() -> None:
    readme = Path("README.md").read_text()

    assert "Python 3.12" in readme
    assert "uv sync --python 3.12 --extra dev" in readme
