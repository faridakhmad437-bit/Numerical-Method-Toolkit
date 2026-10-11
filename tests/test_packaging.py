import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=SRC,
        check=False,
    )


def test_package_imports_without_optional_dev_dependencies():
    # Only numpy is a runtime dependency; block the dev-only packages.
    result = run_python(
        "import sys\n"
        "for name in ('matplotlib', 'scipy', 'pytest'):\n"
        "    sys.modules[name] = None\n"
        "import numerical_toolkit\n"
    )

    assert result.returncode == 0, result.stderr


def test_package_version_matches_project_metadata():
    import numerical_toolkit

    pyproject = (SRC.parent / "pyproject.toml").read_text(encoding="utf-8")

    assert f'version="{numerical_toolkit.__version__}"' in pyproject.replace(" ", "")


def test_project_declares_license_file():
    root = SRC.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert (root / "LICENSE").is_file()
    assert 'license="MIT"' in pyproject.replace(" ", "")
