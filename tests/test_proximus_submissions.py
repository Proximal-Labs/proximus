"""Unit tests for the Submissions subsystem: repository setup, commit
pinning, index computation, and the ``submissions.jsonl`` record."""

import json
import logging
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult

from harbor_proximus.submissions import (
    DEFAULT_SUBMISSION_TIMEOUT_SEC,
    SubmissionError,
    Submissions,
)
from harbor_proximus.tools import EnvironmentGoneError

pytestmark = pytest.mark.unit

logger = logging.getLogger(__name__)

RECORD_HASH = "a1b2c3d4e5f6" + "0" * 28
RECORD_OK = ExecResult(
    stdout=f"subject=fix the bug\nhash={RECORD_HASH}", stderr=None, return_code=0
)


def record_ok(n: int) -> ExecResult:
    return ExecResult(
        stdout=f"subject=step {n}\nhash={n + 1:040x}", stderr=None, return_code=0
    )


RECORD_BAD_REF = ExecResult(
    stdout=None,
    stderr=(
        "'badref' is not a commit in the workspace repository. "
        "Pass a valid commit hash, or omit commit to snapshot "
        "the current workspace."
    ),
    return_code=6,
)


class FakeEnv:
    """Duck-typed environment capturing exec calls."""

    def __init__(self, exec_results: list[ExecResult] | None = None):
        self.exec_calls: list[dict] = []
        self._exec_results = list(exec_results or [])

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout_sec": timeout_sec,
                "env": env,
                "user": user,
            }
        )
        if self._exec_results:
            return self._exec_results.pop(0)
        return RECORD_OK


class TimeoutEnv(FakeEnv):
    async def exec(self, *args, **kwargs):
        raise TimeoutError()


class GoneEnv(FakeEnv):
    async def exec(self, *args, **kwargs):
        raise EnvironmentGoneError("Modal sandbox is gone")


def make_submissions(
    logs_dir: Path,
    env: FakeEnv | None = None,
    *,
    workdir: str = "/app",
    elapsed: float = 12.34,
    submission_timeout_sec: int = DEFAULT_SUBMISSION_TIMEOUT_SEC,
) -> Submissions:
    return Submissions(
        logs_dir,
        env if env is not None else FakeEnv(),  # type: ignore[arg-type]
        workdir=workdir,
        elapsed_sec=lambda: elapsed,
        logger=logger,
        submission_timeout_sec=submission_timeout_sec,
    )


# Injected by Submissions._git_exec on EVERY harness git call: root git refuses the
# agent-owned repo without safe.directory.
def git_safe_env(wd: str = "/app") -> dict:
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": wd,
    }


# ----- the record (submissions.jsonl + summary/winner) -----


async def test_submit_records_row_to_jsonl(tmp_path: Path):
    subs = make_submissions(tmp_path)
    await subs.submit("HEAD")

    lines = (tmp_path / "submissions.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event"] == "submission"
    assert row["index"] == 1
    assert row["ok"] is True
    assert row["hash"] == RECORD_HASH
    assert row["description"] == "fix the bug"
    assert row["elapsed_sec"] == 12.3
    assert "ts" in row
    # The script pinned an existing commit; the harness made none.
    assert "harness_commit" not in row


async def test_harness_made_commit_is_flagged_in_the_row(tmp_path: Path):
    # A commitless submit on a dirty tree makes the record script commit
    # the live workspace (`committed=1` on stdout); the row must say the
    # harness made that commit — flagged like ``auto``, present only when
    # true — so analysis can tell it from a model-made commit.
    snapshot = ExecResult(
        stdout=(
            f"subject=Proximus submission snapshot\nhash={RECORD_HASH}\ncommitted=1"
        ),
        stderr=None,
        return_code=0,
    )
    subs = make_submissions(tmp_path, FakeEnv([snapshot]))
    submission = await subs.submit("")

    assert submission.harness_committed is True
    row = json.loads((tmp_path / "submissions.jsonl").read_text())
    assert row["ok"] is True
    assert row["hash"] == RECORD_HASH
    assert row["harness_commit"] is True
    (summary_row,) = subs.summary()
    assert summary_row["harness_commit"] is True


async def test_failed_record_recorded_with_error(tmp_path: Path):
    subs = make_submissions(tmp_path, FakeEnv([RECORD_BAD_REF]))
    with pytest.raises(SubmissionError):
        await subs.submit("badref")

    (row,) = subs.summary()
    assert row["ok"] is False
    assert "not a commit" in row["error"]
    assert subs.success_count() == 0


async def test_summary_indexes_all_submissions_in_order(tmp_path: Path):
    subs = make_submissions(tmp_path, FakeEnv([record_ok(n) for n in range(3)]))
    for _ in range(3):
        await subs.submit("HEAD")

    rows = subs.summary()
    assert [r["index"] for r in rows] == [1, 2, 3]
    assert [r["hash"] for r in rows] == [f"{n:040x}" for n in (1, 2, 3)]
    assert all("event" not in r for r in rows)
    assert len((tmp_path / "submissions.jsonl").read_text().splitlines()) == 3


async def test_winner_hash_is_newest_successful_recording(tmp_path: Path):
    subs = make_submissions(
        tmp_path, FakeEnv([record_ok(0), record_ok(1), RECORD_BAD_REF])
    )
    assert subs.winner_hash() is None
    await subs.submit("HEAD")
    await subs.submit("HEAD")
    # A failed recording after a good one must not displace the winner.
    with pytest.raises(SubmissionError):
        await subs.submit("badref")
    assert subs.winner_hash() == f"{2:040x}"
    assert subs.success_count() == 2


async def test_unwritable_record_path_never_raises(tmp_path: Path):
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("file where the dir should be")
    subs = make_submissions(blocked / "logs")
    await subs.submit("HEAD")  # must not raise
    assert len(subs.summary()) == 1


# ----- the git recording -----


async def test_submit_resolves_and_pins_the_ref(tmp_path: Path):
    env = FakeEnv([RECORD_OK])
    submission = await make_submissions(tmp_path, env).submit("HEAD")
    assert submission.hash == RECORD_HASH
    assert submission.description == "fix the bug"
    assert submission.duration_sec >= 0
    call = env.exec_calls[0]
    command = call["command"]
    # Script inputs travel as exec env vars, never spliced into the script.
    assert call["env"] == {**git_safe_env(), "wd": "/app", "commit": "HEAD", "idx": "1"}
    # The ref is resolved to a full SHA NOW (HEAD moves, short hashes and
    # branch names drift) and pinned under refs/proximus/submissions/ so
    # a later rebase/reset or `git gc` cannot make it unreachable before
    # the trial's end-of-run snapshot.
    assert 'git rev-parse -q --verify "$commit^{commit}"' in command
    assert 'git update-ref "refs/proximus/submissions/$idx"' in command
    # Nothing is copied: no snapshot dir, no rsync, no archive.
    assert "rsync" not in command
    assert "git archive" not in command
    assert call["timeout_sec"] == DEFAULT_SUBMISSION_TIMEOUT_SEC
    assert call["user"] == "root"


async def test_submit_honors_configured_ten_minute_timeout(tmp_path: Path):
    env = FakeEnv([RECORD_OK])
    await make_submissions(tmp_path, env, submission_timeout_sec=600).submit("HEAD")
    assert env.exec_calls[0]["timeout_sec"] == 600


async def test_submit_passes_the_model_ref_out_of_band(tmp_path: Path):
    # The model-controlled ref travels as an exec env var, never spliced
    # into the script text — there is nothing to shell-quote.
    env = FakeEnv([record_ok(1), RECORD_OK])
    subs = make_submissions(tmp_path, env)
    await subs.submit("HEAD")
    await subs.submit("bad; rm -rf /")
    call = env.exec_calls[1]
    assert call["env"]["commit"] == "bad; rm -rf /"
    assert call["env"]["idx"] == "2"
    assert "rm -rf" not in call["command"]


async def test_failed_recording_does_not_consume_an_index(tmp_path: Path):
    # The pin index counts successful recordings only: after a failure the
    # next success is still submission #1, matching the model-facing note.
    env = FakeEnv([RECORD_BAD_REF, RECORD_OK])
    subs = make_submissions(tmp_path, env)
    with pytest.raises(SubmissionError):
        await subs.submit("badref")
    await subs.submit("HEAD")
    assert env.exec_calls[0]["env"]["idx"] == "1"
    assert env.exec_calls[1]["env"]["idx"] == "1"
    assert subs.success_count() == 1


async def test_submit_unknown_ref_reports_model_readable_guidance(tmp_path: Path):
    subs = make_submissions(tmp_path, FakeEnv([RECORD_BAD_REF]))
    with pytest.raises(SubmissionError) as excinfo:
        await subs.submit("badref")
    assert "not a commit" in str(excinfo.value)
    assert "omit commit to snapshot" in str(excinfo.value)


async def test_submit_malformed_output_still_succeeds(tmp_path: Path):
    # A zero exit means the ref resolved and was pinned; a malformed
    # output must degrade to empty fields, never to a failed submission.
    env = FakeEnv([ExecResult(stdout="something odd", stderr=None, return_code=0)])
    submission = await make_submissions(tmp_path, env).submit("HEAD")
    assert submission.hash == ""
    assert submission.description == ""


async def test_submit_timeout_reports_detail(tmp_path: Path):
    with pytest.raises(SubmissionError, match="timed out"):
        await make_submissions(tmp_path, TimeoutEnv()).submit("HEAD")


async def test_submit_env_gone_propagates_and_records_nothing(tmp_path: Path):
    # A dead sandbox must end the run, not read as a failed submission —
    # and nothing lands in the record for the aborted call.
    subs = make_submissions(tmp_path, GoneEnv())
    with pytest.raises(EnvironmentGoneError):
        await subs.submit("HEAD")
    assert subs.summary() == []


async def test_submit_sandbox_timeout_propagates_and_records_nothing(tmp_path: Path):
    # Modal's SandboxTimeoutError subclasses builtin TimeoutError, so the
    # recording path's `except TimeoutError` sees it first — it must still
    # classify as a dead sandbox, not record a timed-out failure.
    class SandboxTimeoutEnv(FakeEnv):
        async def exec(self, *args, **kwargs):
            raise type(
                "SandboxTimeoutError",
                (TimeoutError,),
                {"__module__": "modal.exception"},
            )("sandbox lifetime expired")

    subs = make_submissions(tmp_path, SandboxTimeoutEnv())
    with pytest.raises(EnvironmentGoneError):
        await subs.submit("HEAD")
    assert subs.summary() == []


# ----- repository setup -----


async def test_ensure_git_passes_wd_out_of_band(tmp_path: Path):
    env = FakeEnv([ExecResult(stdout="initialized\n", stderr=None, return_code=0)])
    subs = make_submissions(tmp_path, env)
    await subs.ensure_git()
    call = env.exec_calls[0]
    assert call["env"] == {**git_safe_env(), "wd": "/app", "exclude_patterns": ""}
    assert "is-inside-work-tree" in call["command"]
    assert call["user"] == "root"


async def test_git_exec_safe_directory_follows_the_workdir(tmp_path: Path):
    # The safe.directory value must be THE workspace, not a hardcoded /app.
    env = FakeEnv([ExecResult(stdout="ok\n", stderr=None, return_code=0)])
    subs = make_submissions(tmp_path, env, workdir="/workspace/repo")
    await subs.ensure_git()
    assert env.exec_calls[0]["env"]["GIT_CONFIG_VALUE_0"] == "/workspace/repo"
    assert env.exec_calls[0]["env"]["GIT_CONFIG_KEY_0"] == "safe.directory"


async def test_matches_winner_uses_root_and_passes_inputs(tmp_path: Path):
    env = FakeEnv([ExecResult(stdout="matched\n", stderr=None, return_code=0)])
    subs = make_submissions(tmp_path, env)
    assert await subs.matches_winner("", "abc123") is True
    call = env.exec_calls[0]
    assert call["user"] == "root"
    assert call["env"] == {
        **git_safe_env(),
        "wd": "/app",
        "winner": "abc123",
        "commit": "",
    }
    assert 'echo "matched"' in call["command"]


async def test_matches_winner_false_on_no_match(tmp_path: Path):
    env = FakeEnv([ExecResult(stdout=None, stderr=None, return_code=1)])
    subs = make_submissions(tmp_path, env)
    assert await subs.matches_winner("fix2", "abc123") is False


async def test_ensure_git_raises_when_workspace_cannot_host_a_repo(tmp_path: Path):
    # A submission IS a commit and submit is the only voluntary exit, so a
    # workspace without git must fail the trial at setup, not let the run
    # burn its budget toward an impossible submit.
    env = FakeEnv(
        [
            ExecResult(
                stdout=None,
                stderr="git is not installed in the task image",
                return_code=7,
            )
        ]
    )
    with pytest.raises(RuntimeError, match="not installed"):
        await make_submissions(tmp_path, env).ensure_git()


async def test_ensure_git_env_gone_propagates(tmp_path: Path):
    # Setup fail-fast: a dead sandbox propagates as-is (the trial errors
    # with the real cause) instead of degrading to a status string.
    with pytest.raises(EnvironmentGoneError):
        await make_submissions(tmp_path, GoneEnv()).ensure_git()


async def test_valid_hash_count_ignores_empty_and_abbreviated_hashes(tmp_path: Path):
    subs = make_submissions(
        tmp_path,
        FakeEnv(
            [
                RECORD_OK,
                ExecResult(stdout="hash=deadbeef", stderr=None, return_code=0),
                ExecResult(stdout="unexpected output", stderr=None, return_code=0),
            ]
        ),
    )

    await subs.submit("HEAD")
    await subs.submit("HEAD")
    await subs.submit("HEAD")

    assert subs.success_count() == 3
    assert subs.valid_hash_count() == 1
    assert subs.winner_hash() == RECORD_HASH
