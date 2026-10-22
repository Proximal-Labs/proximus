"""Install a built wheel outside the source tree and run its offline suite.

Usage: python scripts/verify_wheel.py dist/harbor_proximus-*.whl

Dependency installation uses the network/cache. Harbor's litellm requirement is
overridden away, as in pyproject.toml, and the script checks it stays absent.
Tests block network access and use fake providers. Requires uv on PATH;
temporary environments are removed.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="proximus-wheel-") as directory:
        workdir = Path(directory)
        environment = workdir / "venv"
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)], check=True
        )
        dependencies = [
            str(wheel),
            "pytest>=8.4.2",
            "pytest-asyncio>=1.2",
            "pillow>=10",
        ]
        overrides = workdir / "overrides.txt"
        overrides.write_text("litellm; sys_platform == 'never'\n")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--overrides",
                str(overrides),
                *dependencies,
            ],
            check=True,
        )
        shutil.copytree(
            root / "tests",
            workdir / "tests",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy(root / "pyproject.toml", workdir / "pyproject.toml")
        test_env = dict(os.environ)
        test_env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                str(python),
                "-c",
                "import harbor_proximus, importlib.util; from pathlib import Path; import sys; assert Path(harbor_proximus.__file__).is_relative_to(Path(sys.prefix)); assert importlib.util.find_spec('litellm') is None, 'litellm is installed'; print('Installed package:', harbor_proximus.__file__)",
            ],
            cwd=workdir,
            env=test_env,
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pytest", "-q"], cwd=workdir, env=test_env, check=True
        )


if __name__ == "__main__":
    main()
