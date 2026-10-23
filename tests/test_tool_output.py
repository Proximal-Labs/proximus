"""Large bash output stays available in full outside the model's preview."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.base import ExecResult

from harbor_proximus.tools import (
    BASH_DESCRIPTION,
    OUTPUT_LIMIT_CHARS,
    BashArgs,
    EnvironmentGoneError,
)
from tests.test_proximus_tools import FakeEnv, make_executor


class OutputEnv(FakeEnv):
    def __init__(self, stdout, stderr=None, code=0):
        super().__init__([ExecResult(stdout=stdout, stderr=stderr, return_code=code)])
        self.files = {}

    async def upload_file(self, source_path, target_path):
        source = Path(source_path)
        assert source.stat().st_mode & 0o777 == 0o644
        assert source.parent.stat().st_mode & 0o777 == 0o700
        self.files[target_path] = source.read_bytes()


@pytest.mark.parametrize(
    "stdout,stderr",
    [
        ("START\n" + "x" * 40_000 + "\nEND\n", "warning\n \r\n"),
        (None, "e" * 40_000 + "\n\n"),
        ("a" * 20_000, "b" * 20_000),
        ("界🙂\r\n" * 10_000, "é\n"),
        ("\n" * 40_000, None),
        ("x" * OUTPUT_LIMIT_CHARS, "e"),  # stderr label crosses the threshold
    ],
)
async def test_full_streams_preserved_exactly(stdout, stderr):
    env = OutputEnv(stdout, stderr)
    result = await make_executor(env).bash(BashArgs(command="produce output"))
    assert not result.is_error
    assert len(env.files) == 2
    # Seeing stdout's name must not let a sandbox process preplant a symlink
    # at the next upload's path when the backend writes files as root.
    assert len({path.rsplit(".", 1)[0] for path in env.files}) == 2
    for stream, text in (("stdout", stdout), ("stderr", stderr)):
        path = next(p for p in env.files if p.endswith(f".{stream}"))
        assert path.startswith("/tmp/proximus-output-")
        assert env.files[path] == (text or "").encode("utf-8")
        assert f"Full captured {stream}: {path}" in result.content
    assert "Read ranges or search" in result.content
    assert len(result.content) < OUTPUT_LIMIT_CHARS + 1000


async def test_preview_keeps_head_and_tail_and_files_keep_middle():
    env = OutputEnv("HEAD" + "a" * 20_000 + "MIDDLE" + "b" * 20_000 + "TAIL")
    result = await make_executor(env).bash(BashArgs(command="produce output"))
    assert "HEAD" in result.content and result.content.endswith("TAIL")
    assert "MIDDLE" not in result.content and "output truncated" in result.content
    assert b"MIDDLE" in next(v for p, v in env.files.items() if p.endswith(".stdout"))


@pytest.mark.parametrize("size", [0, 1, OUTPUT_LIMIT_CHARS])
async def test_small_output_needs_no_extra_calls(size):
    env = OutputEnv("x" * size)
    result = await make_executor(env).bash(BashArgs(command="echo small"))
    assert not result.is_error
    assert not env.files
    assert len(env.exec_calls) == 1


async def test_paths_are_unique_across_calls_and_executors():
    env = OutputEnv("a" * 40_000)
    await make_executor(env).bash(BashArgs(command="first"))
    env._exec_results.append(ExecResult(stdout="b" * 40_000, return_code=0))
    await make_executor(env).bash(BashArgs(command="second"))
    assert len(env.files) == 4


@pytest.mark.parametrize("code", [0, 2, -1])
async def test_success_failure_and_timeout_preserve_output(code, monkeypatch):
    ticks = iter([0.0, 600.0])
    monkeypatch.setattr(
        "harbor_proximus.tools.time", SimpleNamespace(monotonic=lambda: next(ticks))
    )
    env = OutputEnv("a" * 40_000, code=code)
    result = await make_executor(env).bash(BashArgs(command="long job"))
    assert len(env.files) == 2
    assert result.is_error == (code != 0)
    if code == -1:
        assert "timed out" in result.content
        assert "may still be running" in result.content
        assert "was killed" not in result.content


@pytest.mark.parametrize("error", [OSError("disk full"), TimeoutError()])
async def test_save_failure_retains_preview_and_warns_without_rerunning(error):
    class BrokenEnv(OutputEnv):
        async def upload_file(self, source_path, target_path):
            if target_path.endswith(".stdout"):
                raise error
            await super().upload_file(source_path, target_path)

    env = BrokenEnv("HEAD" + "a" * 40_000 + "TAIL", "stderr\n")
    result = await make_executor(env).bash(BashArgs(command="train"))
    assert result.is_error
    assert "exit code: 0" in result.content
    assert "WARNING: could not save full stdout" in result.content
    assert "the command already ran" in result.content
    assert "Full captured stdout:" not in result.content
    assert "Full captured stderr:" in result.content
    assert "HEAD" in result.content and result.content.endswith("stderr")
    assert len(env.exec_calls) == 1


@pytest.mark.parametrize(
    "error", [EnvironmentGoneError("gone"), asyncio.CancelledError()]
)
async def test_save_does_not_swallow_environment_loss_or_cancellation(error):
    class BrokenEnv(OutputEnv):
        async def upload_file(self, source_path, target_path):
            raise error

    with pytest.raises(type(error)):
        await make_executor(BrokenEnv("x" * 40_000)).bash(BashArgs(command="long"))


async def test_saved_output_can_be_read_through_later_bash_calls(tmp_path):
    """Exercise actual shell output, file upload and follow-up reads offline."""
    import os
    import shutil

    class LocalEnv:
        def __init__(self):
            self.uploads = []

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            p = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd,
                env={"PATH": os.environ["PATH"], **(env or {})},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                p.communicate(), timeout=timeout_sec
            )
            return ExecResult(
                stdout=stdout.decode(), stderr=stderr.decode(), return_code=p.returncode
            )

        async def upload_file(self, source_path, target_path):
            self.uploads.append(Path(target_path))
            shutil.copy2(source_path, target_path)

    env = LocalEnv()
    executor = make_executor(env, default_workdir=str(tmp_path))
    try:
        result = await executor.bash(
            BashArgs(
                command="printf 'HEAD'; printf 'a%.0s' {1..20000}; printf 'MIDDLE'; "
                "printf 'b%.0s' {1..20000}; printf 'TAIL\\n'; printf 'warning\\n' >&2"
            )
        )
        assert not result.is_error and "MIDDLE" not in result.content
        stdout, stderr = env.uploads
        assert (
            stdout.read_text()
            == "HEAD" + "a" * 20_000 + "MIDDLE" + "b" * 20_000 + "TAIL\n"
        )
        assert stderr.read_text() == "warning\n"
        followup = await executor.bash(
            BashArgs(command=f"grep -o MIDDLE {stdout}; cat {stderr}")
        )
        assert followup.content == "exit code: 0\nMIDDLE\nwarning"
        assert len(env.uploads) == 2
    finally:
        for path in env.uploads:
            path.unlink(missing_ok=True)


def test_description_explains_sequential_calls_background_work_and_timeout():
    for phrase in (
        "sequentially in response order",
        "calls in the same response",
        "separate log and PID files for each job",
        "nohup ./run.sh > /tmp/run.log 2>&1 & echo $! > /tmp/run.pid",
        "Set timeout_sec",
        "does not guarantee the process was killed",
    ):
        assert phrase in BASH_DESCRIPTION


async def test_bash_output_file_prefix_can_be_set():
    env = OutputEnv("x" * 40_000)
    await make_executor(env, bash_output_file_prefix="/tmp/output-").bash(
        BashArgs(command="produce")
    )
    assert len(env.files) == 2 and all(p.startswith("/tmp/output-") for p in env.files)
