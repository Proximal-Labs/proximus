"""Install a built wheel outside the source tree and run its offline suite.

Usage: python scripts/verify_wheel.py dist/harbor_proximus-*.whl
       python scripts/verify_wheel.py dist/harbor_proximus-*.whl --litellm-version 1.97.0

Dependency installation uses the network/cache. Tests block network access and
use fake providers. Requires uv on PATH; temporary environments are removed.
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
    parser.add_argument("--litellm-version")
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
        if args.litellm_version:
            dependencies.append(f"litellm=={args.litellm_version}")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), *dependencies], check=True
        )
        shutil.copytree(
            root / "tests",
            workdir / "tests",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy(root / "pyproject.toml", workdir / "pyproject.toml")
        test_env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
        test_env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                str(python),
                "-c",
                "import harbor_proximus; from pathlib import Path; import sys; assert Path(harbor_proximus.__file__).is_relative_to(Path(sys.prefix)); print('Installed package:', harbor_proximus.__file__)",
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
