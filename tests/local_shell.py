import os
import subprocess
from collections.abc import Mapping

from harbor.environments.base import ExecResult


class LocalShellEnv:
    """Duck-typed environment that runs commands for real with bash, the way
    a sandbox would. Output is decoded as strict UTF-8, as Modal's SDK does:
    a split character raises UnicodeDecodeError."""

    def __init__(self, base_env: Mapping[str, str] = os.environ):
        self._base_env = base_env

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            encoding="utf-8",
            env={**self._base_env, **(env or {})},
            cwd=cwd,
            timeout=timeout_sec,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )
