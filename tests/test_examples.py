import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_example_runs():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "IK:" in result.stdout


def test_readme_python_examples_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = [
        block.split("```", 1)[0]
        for block in readme.split("```python\n")[1:]
    ]

    assert blocks

    for block in blocks:
        exec(compile(block, "README.md", "exec"), {})
